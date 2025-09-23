#!/bin/bash

# Script to generate FMHA v2 kernel sources
# Based on the genrule from tensorrt_llm.BUILD

set -e  # Exit on any error

# Set up the environment for SM80-SM100 generation
export ENABLE_SM100=1
export TORCH_CUDA_ARCH_LIST="8.0,8.6,8.9,9.0,10.0"
export GENERATE_CU_TRTLLM="true"
export GENERATE_CUBIN="true"
export ENABLE_HMMA_FP32="true"

# Path to the TensorRT-LLM setup script
SETUP_SCRIPT="/home/philkuz/dev/TensorRT-LLM/cpp/kernels/fmha_v2/setup.py"
SETUP_DIR=$(dirname "$SETUP_SCRIPT")

cd "$SETUP_DIR"
# Run the setup script to generate kernel sources
echo "Running setup.py to generate kernel sources..."
python3 setup.py

# Show what was generated
echo "Generated files:"
find "$SETUP_DIR/generated" -name "*.cu" | head -20
echo "..."
echo "Total .cu files: $(find "$SETUP_DIR/generated" -name "*.cu" | wc -l)"

cd $(git rev-parse --show-toplevel)
cp cpp/kernels/fmha_v2/generated/fmha_cubin.h cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/fmha_cubin.h