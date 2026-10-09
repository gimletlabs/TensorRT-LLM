# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the standalone C++ library build used by Gimlet."""

import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def build_wheel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    script = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRTLLM_BUILD_ROOT", raising=False)
    monkeypatch.setattr(module, "get_project_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "setup_venv", Mock(return_value=(Path("python"), Path("conan"))))
    monkeypatch.setattr(module, "generate_fmha_cu", Mock())
    monkeypatch.setattr(module, "build_run", Mock())
    return module


def test_cpp_only_build_excludes_python_targets(build_wheel: ModuleType, tmp_path: Path) -> None:
    parser = ArgumentParser()
    build_wheel.add_arguments(parser)
    args = parser.parse_args(["--cpp_only", "--cuda_architectures=100-real"])
    args.build_dir = tmp_path / "build"
    build_wheel.main(**vars(args))
    commands = [call.args[0] for call in build_wheel.build_run.call_args_list]
    configure = next(command for command in commands if command.startswith("cmake -D"))
    for target in ("PYT", "DEEP_EP", "DEEP_GEMM", "FLASH_MLA", "NCCL_EXTENSIONS"):
        assert f'-DBUILD_{target}="OFF"' in configure
    assert '-DBUILD_WHEEL_TARGETS="tensorrt_llm"' in configure
    assert any(command.startswith("cmake --build") for command in commands)


def test_setup_only_stops_before_codegen_and_build(build_wheel: ModuleType) -> None:
    parser = ArgumentParser()
    build_wheel.add_arguments(parser)
    build_wheel.main(**vars(parser.parse_args(["--only_run_setup"])))
    build_wheel.setup_venv.assert_called_once()
    build_wheel.generate_fmha_cu.assert_not_called()
    build_wheel.build_run.assert_not_called()
