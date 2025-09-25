#!/bin/bash
set -e

# Set up environment variables
export PATH="/usr/local/cuda/bin:/snap/bin:$PATH"
export PYTHONPATH="/home/philkuz/dev/TensorRT-LLM/3rdparty/cutlass/python:${PYTHONPATH:-}"
export CUDA_CACHE_MAXSIZE=2147483648
export CC=gcc
export CXX=g++
TARDIR=${GML_BUILD_DIR:-/tmp/tensorrt-llm}
TARFILE=${GML_TARFILE:-/tmp/tensorrt-llm.tar.gz}
CREATE_TARFILE=${GML_CREATE_TARFILE:-true}

echo "TARDIR: $TARDIR"
echo "TARFILE: $TARFILE"
echo "CREATE_TARFILE: $CREATE_TARFILE"

# Build the wheel
python3 scripts/build_wheel.py \
    --use_ccache \
    --clean \
    --cpp_only \
    -G Ninja \
    -j 192 \
    -a '80-real;86-real;89-real;90-real;100-real;120-real' \
    --extra-cmake-vars ENABLE_MULTI_DEVICE=0 \
    --extra-cmake-vars ENABLE_UCX=0 \
    --extra-cmake-vars CMAKE_C_COMPILER=gcc \
    --extra-cmake-vars CMAKE_CXX_COMPILER=g++ \
    --extra-cmake-vars CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    --extra-cmake-vars CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

mkdir -p /tmp/tensorrt-llm

cp cpp/build/tensorrt_llm/libtensorrt_llm.so "$TARDIR/"
cp cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so "$TARDIR/"
cp cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so "$TARDIR/"
cp cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so "$TARDIR/"

if [ "$CREATE_TARFILE" = true ]; then
    tar -zcf "$TARFILE" -C "$TARDIR" .
fi

echo "TensorRT-LLM built and copied to $TARFILE"

