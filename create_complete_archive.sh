#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M)
ARCHIVE_NAME="physllm_complete_with_models_${TIMESTAMP}.tar.gz"

echo "Creating complete PhysLLM archive with models and binaries..."
echo "This will include:"
echo "  - Source code"
echo "  - Compiled binaries (target/release)"
echo "  - Model weights (~/models)"
echo "  - All documentation"
echo ""

# Create temporary staging directory
STAGING_DIR="/tmp/physllm_package_${TIMESTAMP}"
mkdir -p ${STAGING_DIR}

echo "1. Copying source code..."
rsync -a --exclude='.git' --exclude='*.tar.gz' \
    ~/physllm/ ${STAGING_DIR}/physllm/

echo "2. Copying model weights..."
if [ -d ~/models ]; then
    mkdir -p ${STAGING_DIR}/models
    rsync -a ~/models/ ${STAGING_DIR}/models/
    echo "   ✓ Models copied ($(du -sh ~/models | cut -f1))"
else
    echo "   ! No models directory found"
fi

echo "3. Copying compiled binaries..."
if [ -d ~/physllm/target/release ]; then
    mkdir -p ${STAGING_DIR}/physllm/target/release
    
    # Copy all example binaries
    cp ~/physllm/target/release/examples/* ${STAGING_DIR}/physllm/target/release/examples/ 2>/dev/null || true
    
    # Copy main binaries
    cp ~/physllm/target/release/physllm-server ${STAGING_DIR}/physllm/target/release/ 2>/dev/null || true
    
    # Copy kernel libraries
    find ~/physllm/target/release/build/rocm-backend-*/out -name "*.a" -o -name "*.o" \
        -exec cp {} ${STAGING_DIR}/physllm/kernels/ \; 2>/dev/null || true
    
    echo "   ✓ Binaries copied"
fi

echo "4. Creating archive..."
cd /tmp
tar -czf ~/${ARCHIVE_NAME} physllm_package_${TIMESTAMP}/

# Cleanup
rm -rf ${STAGING_DIR}

# Show results
ARCHIVE_SIZE=$(du -sh ~/${ARCHIVE_NAME} | cut -f1)
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Archive created: ~/${ARCHIVE_NAME}"
echo "  Size: ${ARCHIVE_SIZE}"
echo ""
echo "Contents:"
tar -tzf ~/${ARCHIVE_NAME} | head -20
echo "  ... ($(tar -tzf ~/${ARCHIVE_NAME} | wc -l) total files)"
echo "═══════════════════════════════════════════════════════════"
echo ""
