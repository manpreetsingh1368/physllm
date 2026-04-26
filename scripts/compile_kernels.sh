#!/bin/bash
echo "🔧 Manual kernel compilation"
echo ""

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIPCC=${ROCM_PATH}/bin/hipcc
OUT_DIR="target/kernel_objects"
mkdir -p "$OUT_DIR"

# Detect GPU architecture
if command -v rocminfo &> /dev/null; then
    ARCH=$(rocminfo 2>/dev/null | grep "Name:" | grep "gfx" | head -1 | awk '{print $2}')
    echo "Detected GPU: ${ARCH:-unknown}"
elif command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    ARCH="sm_80"  # Default A100
fi

ARCH_FLAG="--offload-arch=${ARCH:-gfx942}"
echo "Architecture: $ARCH_FLAG"
echo ""

for kernel in kernels/*.hip; do
    name=$(basename "$kernel" .hip)
    echo -n "  Compiling $name..."
    
    if $HIPCC $ARCH_FLAG -O3 -fPIC -c "$kernel" -o "$OUT_DIR/${name}.o" 2>/dev/null; then
        echo " ✓"
    else
        echo " ✗ (failed)"
    fi
done

# Create static library
ar rcs "$OUT_DIR/libphysllm_kernels.a" "$OUT_DIR"/*.o 2>/dev/null
echo ""
echo "✓ Kernel library: $OUT_DIR/libphysllm_kernels.a"
echo ""
echo "Contents:"
ar t "$OUT_DIR/libphysllm_kernels.a"
