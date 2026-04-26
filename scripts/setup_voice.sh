#!/usr/bin/env bash
# scripts/setup_voice.sh — Download and verify all voice models for PhysLLM
# Run once before using the voice feature.

set -e
MODELS_DIR="${MODELS_DIR:-models}"
WHISPER_MODEL="${WHISPER_MODEL:-small.en}"

echo "═══════════════════════════════════════════════════"
echo "  PhysLLM Voice Setup"
echo "═══════════════════════════════════════════════════"

# ── System dependencies ───────────────────────────────────────────────────────
echo -e "\n[1/5] Checking system dependencies..."

check_cmd() {
    if command -v "$1" &>/dev/null; then echo "  ✓ $1"; else echo "  ✗ $1 — $2"; fi
}

check_cmd "espeak-ng" "sudo apt install espeak-ng"
check_cmd "piper"     "pip install piper-tts"
check_cmd "ffmpeg"    "sudo apt install ffmpeg  (needed for audio conversion)"

# ── Whisper model ─────────────────────────────────────────────────────────────
echo -e "\n[2/5] Downloading Whisper model: ${WHISPER_MODEL}..."
mkdir -p "${MODELS_DIR}/whisper"

WHISPER_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
GGML_NAME="ggml-${WHISPER_MODEL}.bin"
DEST="${MODELS_DIR}/whisper/${GGML_NAME}"

if [ -f "$DEST" ]; then
    echo "  ✓ Already exists: $DEST"
else
    echo "  Downloading ${GGML_NAME} (~$(case $WHISPER_MODEL in
        tiny.en) echo '75MB';; base.en) echo '145MB';; small.en) echo '465MB';;
        medium.en) echo '1.5GB';; large-v3) echo '3GB';; *) echo '?';; esac))..."
    wget -q --show-progress -O "$DEST" "${WHISPER_URL}/${GGML_NAME}"
    echo "  ✓ Downloaded: $DEST"
fi

# ── whisper.cpp build (for whisper-rs) ───────────────────────────────────────
echo -e "\n[3/5] Building whisper.cpp..."
if [ ! -d "vendor/whisper.cpp" ]; then
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp vendor/whisper.cpp
fi

cd vendor/whisper.cpp

# Detect if ROCm is available for GPU acceleration
if command -v hipcc &>/dev/null; then
    echo "  ROCm detected — building with HIP acceleration..."
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    cmake -B build \
        -DWHISPER_HIPBLAS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_COMPILER="${ROCM_PATH}/bin/hipcc" \
        -DAMDGPU_TARGETS="gfx1100;gfx1030;gfx906;gfx90a" \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        2>&1 | tail -3
else
    echo "  Building CPU-only whisper.cpp..."
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        2>&1 | tail -3
fi

cmake --build build --config Release -j"$(nproc)" 2>&1 | tail -5
echo "  ✓ whisper.cpp built"
cd ../..

# ── Silero VAD model ──────────────────────────────────────────────────────────
echo -e "\n[4/5] Downloading Silero VAD model..."
mkdir -p "${MODELS_DIR}"
VAD_DEST="${MODELS_DIR}/silero_vad.onnx"

if [ -f "$VAD_DEST" ]; then
    echo "  ✓ Already exists: $VAD_DEST"
else
    wget -q --show-progress \
        -O "$VAD_DEST" \
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    echo "  ✓ Silero VAD downloaded (~1MB)"
fi

# ── Piper TTS voice ───────────────────────────────────────────────────────────
echo -e "\n[5/5] Downloading Piper TTS voice..."
mkdir -p "${MODELS_DIR}/piper"
PIPER_MODEL="en_GB-alan-medium"
PIPER_DEST="${MODELS_DIR}/piper/${PIPER_MODEL}.onnx"

if [ -f "$PIPER_DEST" ]; then
    echo "  ✓ Already exists: $PIPER_DEST"
else
    BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium"
    wget -q --show-progress -O "$PIPER_DEST"       "${BASE}/${PIPER_MODEL}.onnx"
    wget -q --show-progress -O "${PIPER_DEST}.json" "${BASE}/${PIPER_MODEL}.onnx.json"
    echo "  ✓ Piper voice downloaded"
fi

# ── ONNX Runtime (for Silero VAD) ─────────────────────────────────────────────
echo -e "\nChecking ONNX Runtime..."
ORT_LIB="/usr/lib/libonnxruntime.so"
if [ -f "$ORT_LIB" ]; then
    echo "  ✓ ONNX Runtime found: $ORT_LIB"
else
    echo "  Installing ONNX Runtime..."
    ORT_VER="1.18.1"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}"
    wget -q --show-progress \
        -O /tmp/ort.tgz \
        "${ORT_URL}/onnxruntime-linux-x64-${ORT_VER}.tgz"
    tar -xzf /tmp/ort.tgz -C /tmp/
    sudo cp /tmp/onnxruntime-linux-x64-${ORT_VER}/lib/libonnxruntime.so* /usr/lib/
    sudo ldconfig
    rm -f /tmp/ort.tgz
    echo "  ✓ ONNX Runtime installed"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "\n═══════════════════════════════════════════════════"
echo "  Voice setup complete!"
echo ""
echo "  Models:"
echo "    Whisper:    ${MODELS_DIR}/whisper/${GGML_NAME}"
echo "    Silero VAD: ${MODELS_DIR}/silero_vad.onnx"
echo "    Piper TTS:  ${MODELS_DIR}/piper/${PIPER_MODEL}.onnx"
echo ""
echo "  Start the server:"
echo "    cargo run --release --features rocm -p api-server"
echo ""
echo "  Then open the voice client in your browser:"
echo "    firefox scripts/voice_client.html"
echo ""
echo "  Or use voice from the command line:"
echo "    cargo run --release --features rocm --bin physllm-voice"
echo "═══════════════════════════════════════════════════"
