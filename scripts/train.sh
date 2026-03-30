#!/usr/bin/env bash
# scripts/train.sh — Launch PhysLLM training on AMD GPU
# Usage: ./scripts/train.sh [--model 7b|13b] [--data /path/to/data] [--resume]

#!/usr/bin/env bash
# scripts/train.sh — Launch PhysLLM training on AMD GPU(s)
# Supports automatic multi-GPU detection

set -e

MODEL="7b"
DATA_DIR="data/physics_corpus"
RESUME=""
OUTPUT_DIR="checkpoints/physllm-${MODEL}"
LOG_DIR="logs"
BATCH_SIZE=4
GRAD_ACCUM=8
LR=2e-4
WARMUP_STEPS=1000
MAX_STEPS=100000

# --- Parse CLI args ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)   MODEL="$2"; shift ;;
        --data)    DATA_DIR="$2"; shift ;;
        --resume)  RESUME="--resume" ;;
        --batch)   BATCH_SIZE="$2"; shift ;;
        --lr)      LR="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export RUST_LOG="physllm=info,rocm_backend=debug"
export RUST_BACKTRACE=1

# --- Auto-detect GPUs ---
GPU_IDS=$(rocm-smi --showid 2>/dev/null | grep -oE '[0-9]+')
if [[ -z "$GPU_IDS" ]]; then
    echo "Warning: No AMD GPUs detected. Training will run on CPU."
    export HIP_VISIBLE_DEVICES=""
else
    export HIP_VISIBLE_DEVICES=$(echo $GPU_IDS | tr ' ' ',')
fi

GPU_COUNT=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

GPU_NAMES=$(rocm-smi --showproductname 2>/dev/null | grep 'Card' | head -"$GPU_COUNT" | tr '\n' ',' | sed 's/,$//')

echo "PhysLLM Training"
echo "  Model:      physllm-$MODEL"
echo "  Data:       $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  GPUs:       $GPU_NAMES ($GPU_COUNT detected)"
echo "  Batch:      $BATCH_SIZE × $GRAD_ACCUM grad_accum = $(( BATCH_SIZE * GRAD_ACCUM )) effective"
echo "  LR:         $LR"
echo ""

# --- Build optimized training binary ---
cargo build --release --features rocm --bin physllm-train 2>&1 | tail -5

# --- Launch training ---
ROCM_PATH="$ROCM_PATH" \
cargo run --release --features rocm --bin physllm-train -- \
    --config "configs/physllm_${MODEL}.json" \
    --data   "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --batch  "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr     "$LR" \
    --warmup "$WARMUP_STEPS" \
    --steps  "$MAX_STEPS" \
    --gpus   "$GPU_COUNT" \
    --log    "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).jsonl" \
    $RESUME \
    2>&1 | tee "$LOG_DIR/last_run.log"