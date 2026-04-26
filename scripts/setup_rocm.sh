#!/usr/bin/env bash
# scripts/setup_rocm.sh — Verify and configure ROCm for PhysLLM
# Tested on: Ubuntu 22.04, ROCm 6.x, RX 7900 XTX / RX 7900 XT / RX 6900 XT / MI250

set -e
ROCM_VERSION="6.2.0"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo ""
echo "  PhysLLM ROCm Setup Checker"
echo ""

#  1. Detect AMD GPU 
echo -e "\n[1/6] Detecting AMD GPU..."
if ! command -v rocm-smi &>/dev/null; then
    echo "  ✗ rocm-smi not found. Install ROCm:"
    echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

rocm-smi --showproductname 2>/dev/null || true
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
echo "  ✓ GPU architecture: $GPU_ARCH"

#  2. Check ROCm version 
echo -e "\n[2/6] Checking ROCm version..."
INSTALLED_VER=$(cat "$ROCM_PATH/.info/version" 2>/dev/null || echo "unknown")
echo "  Installed: $INSTALLED_VER"
echo "  Required:  $ROCM_VERSION+"

#  3. Check hipcc 
echo -e "\n[3/6] Checking hipcc compiler..."
if ! command -v "$ROCM_PATH/bin/hipcc" &>/dev/null; then
    echo "  ✗ hipcc not found at $ROCM_PATH/bin/hipcc"
    exit 1
fi
echo "  ✓ hipcc: $("$ROCM_PATH/bin/hipcc" --version 2>&1 | head -1)"

#  4. Check rocBLAS / hipBLAS 
echo -e "\n[4/6] Checking GPU math libraries..."
for lib in libamdhip64 librocblas libhipblas libmiopen; do
    if ls "$ROCM_PATH/lib/${lib}.so"* &>/dev/null; then
        echo "  ✓ $lib found"
    else
        echo "  ✗ $lib NOT found — install rocm-libs package"
    fi
done

#  5. Set environment variables 
echo -e "\n[5/6] Environment variables..."
export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION="${GPU_ARCH#gfx}"  # e.g. "1100" for gfx1100

# Write to ~/.cargo/config.toml for build
mkdir -p ~/.cargo
cat >> ~/.cargo/config.toml <<EOF

[env]
ROCM_PATH = "$ROCM_PATH"
HIP_PATH   = "$ROCM_PATH"
HSA_OVERRIDE_GFX_VERSION = "${GPU_ARCH#gfx}"
EOF

echo "  ✓ Environment configured"
echo "  ✓ HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

#  6. Test compile a simple HIP program 
echo -e "\n[6/6] Compiling test HIP program..."
TMPFILE=$(mktemp /tmp/hip_test_XXXX.hip)
cat > "$TMPFILE" << 'HIP'
#include <hip/hip_runtime.h>
#include <stdio.h>
__global__ void hello() { printf("HIP kernel running on GPU %d\n", blockIdx.x); }
int main() {
    hello<<<1,1>>>();
    hipDeviceSynchronize();
    return 0;
}
HIP
if "$ROCM_PATH/bin/hipcc" --amdgpu-target="$GPU_ARCH" -O2 "$TMPFILE" -o /tmp/hip_test 2>/dev/null; then
    /tmp/hip_test 2>/dev/null && echo "  ✓ HIP kernel executed successfully" || echo "  ⚠ Compiled but GPU execution failed (may need permissions)"
else
    echo "  ✗ HIP compilation failed — check ROCm installation"
fi
rm -f "$TMPFILE" /tmp/hip_test

echo -e ""
echo "  Setup complete! Build PhysLLM with:"
echo "    export ROCM_PATH=$ROCM_PATH"
echo "    cargo build --release --features rocm"
echo ""
