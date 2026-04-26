#!/bin/bash
set -e
echo "🔥 Building PhysLLM for NVIDIA GPUs (CUDA via HIP)"
echo ""

export HIP_PLATFORM=nvidia
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=${ROCM_PATH}/hip
export PATH=${CUDA_PATH}/bin:$PATH

# Check nvcc
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. Install CUDA toolkit."
    exit 1
fi

# Check HIP is installed
if ! command -v ${ROCM_PATH}/bin/hipcc &> /dev/null; then
    echo "❌ hipcc not found. Install HIP for NVIDIA:"
    echo "   sudo apt-get install hip-runtime-nvidia hip-dev"
    exit 1
fi

echo "CUDA: $(nvcc --version 2>&1 | tail -1)"
echo ""

cargo build --release --features cuda -p rocm-backend
echo "✓ GPU backend compiled"

cargo build --release -p llm-core
echo "✓ LLM core compiled"

echo ""
echo ""
echo "✓ NVIDIA build complete!"
echo ""
