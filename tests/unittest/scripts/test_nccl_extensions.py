# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CPU regressions for NCCL-EP dependency configuration."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

MODULE = Path(__file__).resolve().parents[3] / "cpp/cmake/modules/prepare_nccl_for_ep.cmake"


def prepare_nccl(
    build: Path, include: Path, library: Path, version: str
) -> subprocess.CompletedProcess[str]:
    build.mkdir(exist_ok=True)
    script = build / "prepare.cmake"
    script.write_text(f'include("{MODULE}")\nprepare_nccl_for_ep()\n')
    return subprocess.run(
        [
            "cmake",
            f"-DNCCL_INCLUDE_DIR={include}",
            f"-DNCCL_LIBRARY={library}",
            f"-DNCCL_VERSION_STRING={version}",
            "-P",
            str(script),
        ],
        cwd=build,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "libdir", ["lib", "lib64", "lib/x86_64-linux-gnu", "lib/aarch64-linux-gnu"]
)
def test_nccl_ep_uses_selected_headers_and_library(tmp_path: Path, libdir: str) -> None:
    compiler = shutil.which("cc")
    assert compiler is not None, "A host C compiler is required for this build regression"
    include = tmp_path / "NCCL installation" / "include"
    include.mkdir(parents=True)
    library = tmp_path / "NCCL installation" / libdir / "libnccl.so"
    library.parent.mkdir(parents=True)
    (include / "nccl.h").write_text("#define NCCL_VERSION_CODE 23007\nint ncclGetVersion(int*);\n")
    source = tmp_path / "nccl.c"
    source.write_text("int ncclGetVersion(int* version) { *version = 23007; return 0; }\n")
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(library)], check=True)

    build = tmp_path / "build"
    result = prepare_nccl(build, include, library, "2.30.7")
    assert result.returncode == 0, result.stdout + result.stderr
    home = build / "nccl"
    assert (home / "include").resolve() == include
    assert (home / "lib/libnccl.so").resolve() == library

    # Use the include/lib layout consumed by upstream NCCL-EP and verify the
    # executable sees the selected header and the selected shared library.
    source.write_text(
        "#include <nccl.h>\n"
        "int main(void) { int version; return ncclGetVersion(&version) || "
        "version != NCCL_VERSION_CODE; }\n"
    )
    executable = tmp_path / "check_nccl"
    subprocess.run(
        [
            compiler,
            str(source),
            f"-I{home / 'include'}",
            f"-L{home / 'lib'}",
            f"-Wl,-rpath,{home / 'lib'}",
            "-lnccl",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)

    # Reconfiguration must replace the installation view, not keep stale links.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "nccl.h").write_text("// replacement headers\n")
    replacement_library = replacement / "libnccl.so"
    shutil.copyfile(library, replacement_library)
    result = prepare_nccl(build, replacement, replacement_library, "2.30.7")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (home / "include").resolve() == replacement
    assert (home / "lib/libnccl.so").resolve() == replacement_library


@pytest.mark.parametrize("version", ["", "2.28.9", "2.29.7", "2.30.6"])
def test_nccl_ep_rejects_incompatible_headers(tmp_path: Path, version: str) -> None:
    result = prepare_nccl(
        tmp_path / "build", tmp_path / "include", tmp_path / "libnccl.so", version
    )
    assert result.returncode != 0
    assert "NCCL-EP requires NCCL >= 2.30.7" in result.stderr
    assert "NCCL_ROOT" in result.stderr


def test_nccl_ep_rejects_missing_installation(tmp_path: Path) -> None:
    result = prepare_nccl(
        tmp_path / "build", tmp_path / "include", tmp_path / "libnccl.so", "2.30.7"
    )
    assert result.returncode != 0
    assert "headers and library to exist" in result.stderr
