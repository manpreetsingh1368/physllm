#!/bin/bash
set -e
echo "🔥 Building PhysLLM for AMD GPUs (ROCm/HIP)"
echo ""

export HIP_PLATFORM=amd
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=${ROCM_PATH}/hip

# Check hipcc exists
if ! command -v ${ROCM_PATH}/bin/hipcc &> /dev/null; then
    echo "❌ hipcc not found at ${ROCM_PATH}/bin/hipcc"
    echo "   Install ROCm: https://rocm.docs.amd.com"
    exit 1
fi

echo "ROCm: $(${ROCM_PATH}/bin/hipcc --version 2>&1 | head -1)"
echo ""

cargo build --release --features rocm -p rocm-backend
echo "✓ GPU backend compiled"

cargo build --release -p llm-core
echo "✓ LLM core compiled"

echo ""
echo ""
echo "✓ AMD build complete!"
echo ""
echo ""
echo "Next steps:"
echo "  1. Download model:  ./scripts/download_model.sh"
echo "  2. Run inference:   cargo run --release -p llm-core --example gpu_inference"
