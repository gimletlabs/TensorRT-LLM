#!/bin/bash

# Set up environment variables
export PATH="/usr/local/cuda/bin:/snap/bin:$PATH"
export PYTHONPATH="/home/philkuz/dev/TensorRT-LLM/3rdparty/cutlass/python:${PYTHONPATH:-}"
export CUDA_CACHE_MAXSIZE=2147483648
export CC=gcc
export CXX=g++

# Build the wheel
python3 scripts/build_wheel.py \
    --use_ccache \
    --clean \
    -G Ninja \
    -j 192 \
    -a '80-real;86-real;89-real;90-real;100-real;120-real' \
    --extra-cmake-vars ENABLE_MULTI_DEVICE=0 \
    --extra-cmake-vars ENABLE_UCX=0 \
    --extra-cmake-vars CMAKE_C_COMPILER=gcc \
    --extra-cmake-vars CMAKE_CXX_COMPILER=g++ \
    --extra-cmake-vars CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    --extra-cmake-vars CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Create directory and unzip (you'll need to specify what to unzip)
mkdir -p /tmp/tensorrt-llm-wheel
# TODO: Add unzip command here - please specify what file to unzip
unzip build/tensorrt_llm-*.whl -d /tmp/tensorrt-llm-wheel

# Create target directory and copy .so files
mkdir -p ~/dev/tensorrt_llm_x86_64
fdfind '.so$' /tmp/tensorrt-llm-wheel/ --exec cp {} ~/dev/tensorrt_llm_x86_64/

echo "Build and copy completed!"
