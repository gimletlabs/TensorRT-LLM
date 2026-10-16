# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(prepare_nccl_for_ep)
  if(NOT NCCL_VERSION_STRING OR NCCL_VERSION_STRING VERSION_LESS "2.30.7")
    message(
      FATAL_ERROR
        "NCCL-EP requires NCCL >= 2.30.7; found '${NCCL_VERSION_STRING}'. Headers: '${NCCL_INCLUDE_DIR}'; library: '${NCCL_LIBRARY}'. Install matching NCCL development/runtime packages, or configure a clean build with -DNCCL_ROOT=<compatible-installation>."
    )
  endif()

  if(NOT EXISTS "${NCCL_INCLUDE_DIR}/nccl.h" OR NOT EXISTS "${NCCL_LIBRARY}")
    message(
      FATAL_ERROR
        "NCCL-EP requires the discovered NCCL headers and library to exist.")
  endif()

  # NCCL-EP expects an include/ and lib/ tree under one NCCL_HOME. Package
  # installations can keep the headers and library in separate directories,
  # including lib64/ or lib/<architecture>-linux-gnu/.
  set(NCCL_HOME
      "${CMAKE_CURRENT_BINARY_DIR}/nccl"
      CACHE PATH "NCCL installation view used by nccl-extensions" FORCE)
  file(MAKE_DIRECTORY "${NCCL_HOME}/lib")
  file(CREATE_LINK "${NCCL_INCLUDE_DIR}" "${NCCL_HOME}/include" SYMBOLIC)
  file(CREATE_LINK "${NCCL_LIBRARY}" "${NCCL_HOME}/lib/libnccl.so" SYMBOLIC)
endfunction()
