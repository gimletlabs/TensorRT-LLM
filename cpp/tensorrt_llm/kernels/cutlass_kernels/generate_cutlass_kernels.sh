#!/bin/bash

# Script to generate Cutlass kernel instantiations
# Based on the CMakeLists.txt process in this directory

set -e  # Exit on any error

# Directory paths
SCRIPT_DIR=$(dirname "$(realpath "$0")")
CUTLASS_DIR="/home/philkuz/dev/TensorRT-LLM/3rdparty/cutlass"
CUTLASS_PYTHON_DIR="$CUTLASS_DIR/python"
GENERATION_PYTHON_DIR="$SCRIPT_DIR/python"
OUTPUT_DIR="$SCRIPT_DIR/cutlass_instantiations"

# CUDA architectures to target (based on our BUILD file configuration)
CUDA_ARCHITECTURES="80;86;89;90;100"

echo "Setting up Cutlass library..."
# Step 1: Setup the cutlass library
cd "$CUTLASS_PYTHON_DIR"

# Set up PYTHONPATH to include the cutlass python directory
export PYTHONPATH="$CUTLASS_PYTHON_DIR:$PYTHONPATH"

# Install cutlass library locally
python3 setup_library.py develop --user --prefix="$HOME/.local"

if [ $? -ne 0 ]; then
    echo "Warning: cutlass library setup had issues, but continuing..."
fi

echo "Generating Cutlass kernel instantiations..."
# Step 2: Generate kernel instantiations
cd "$GENERATION_PYTHON_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Make sure PYTHONPATH includes cutlass library
export PYTHONPATH="$CUTLASS_PYTHON_DIR:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH"

python3 generate_kernels.py -a "$CUDA_ARCHITECTURES" -o "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Failed to generate CUTLASS kernel instantiations"
    exit 1
fi

echo "Successfully generated Cutlass kernels in: $OUTPUT_DIR"

# Show what was generated
echo "Generated directories:"
find "$OUTPUT_DIR" -type d | head -10
echo ""
echo "Generated .cu files:"
find "$OUTPUT_DIR" -name "*.cu" | wc -l
echo "files generated"

echo ""
echo "Sample generated files:"
find "$OUTPUT_DIR" -name "*.cu" | head -5
