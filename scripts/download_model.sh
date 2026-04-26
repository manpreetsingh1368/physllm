#!/bin/bash
echo "📦 Downloading Mistral 7B v0.1..."
echo ""

MODEL_DIR="${1:-models/mistral-7b}"
mkdir -p "$MODEL_DIR"

pip install -q huggingface-hub 2>/dev/null

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'mistralai/Mistral-7B-v0.1',
    local_dir='${MODEL_DIR}',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'],
)
print('✓ Model downloaded to ${MODEL_DIR}')
"
