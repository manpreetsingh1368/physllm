#!/bin/bash
set -e
echo "Building PhysLLM (CPU-only mode)"
cargo build --release -p rocm-backend
cargo build --release -p llm-core
echo "✓ CPU build complete (no GPU acceleration)"
