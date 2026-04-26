#!/usr/bin/env bash
# scripts/train.sh — Production PhysLLM training launcher
set -euo pipefail

MODEL_ID="mistral-7b"; MODEL_DIR="models/${MODEL_ID}"
TRAIN_DATA="data/train/train.jsonl"; VAL_DATA="data/train/val.jsonl"
OUTPUT_DIR="checkpoints/physllm-7b"
LORA_RANK=32; LORA_ALPHA=64; LORA_DROPOUT=0.05
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LR=2e-4; WARMUP_STEPS=100; TOTAL_STEPS=10000
BATCH_SIZE=2; GRAD_ACCUM=8; MAX_SEQ_LEN=2048
SAVE_EVERY=500; EVAL_EVERY=500; LOG_EVERY=10; KEEP_CKPTS=3
DTYPE="f16"; SEED=42; WANDB_PROJECT="physllm"; RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_ID="$2"; MODEL_DIR="models/$2"; shift 2;;
    --model-dir)   MODEL_DIR="$2"; shift 2;;
    --output)      OUTPUT_DIR="$2"; shift 2;;
    --resume)      RESUME="$2"; shift 2;;
    --lora-rank)   LORA_RANK="$2"; shift 2;;
    --lr)          LR="$2"; shift 2;;
    --warmup)      WARMUP_STEPS="$2"; shift 2;;
    --steps)       TOTAL_STEPS="$2"; shift 2;;
    --batch)       BATCH_SIZE="$2"; shift 2;;
    --grad-accum)  GRAD_ACCUM="$2"; shift 2;;
    --no-wandb)    WANDB_PROJECT=""; shift;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export RUST_LOG="${RUST_LOG:-physllm_train=debug,info}"
export RUST_BACKTRACE=1
export HIP_FORCE_DEV_KERNARG=1

EFF_BATCH=$(( BATCH_SIZE * GRAD_ACCUM ))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════"
echo "  PhysLLM Production Trainer"
echo "  Model:       ${MODEL_DIR}"
echo "  LR:          ${LR}  steps=${TOTAL_STEPS}  warmup=${WARMUP_STEPS}"
echo "  LoRA rank:   ${LORA_RANK}  alpha=${LORA_ALPHA}"
echo "  Eff. batch:  ${EFF_BATCH} = ${BATCH_SIZE} × ${GRAD_ACCUM} grad_accum"
echo "  dtype:       ${DTYPE}"
echo "═══════════════════════════════════════════════════"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "ERROR: ${MODEL_DIR} not found."
  echo "Download: huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ${MODEL_DIR}"
  exit 1
fi

for f in "${TRAIN_DATA}" "${VAL_DATA}"; do
  [ ! -f "$f" ] && echo "ERROR: $f not found. Run: python3 scripts/prepare_data.py" && exit 1
done

echo "Building physllm-train..."
cargo build --release --features rocm --bin physllm-train 2>&1 | tail -3

RESUME_ARG="${RESUME:+--resume ${RESUME}}"

ROCM_PATH="${ROCM_PATH}" \
cargo run --release --features rocm --bin physllm-train -- \
  --model-dir "${MODEL_DIR}" --train-data "${TRAIN_DATA}" --val-data "${VAL_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}" \
  --lora-targets "${LORA_TARGETS}" \
  --lr "${LR}" --warmup-steps "${WARMUP_STEPS}" --total-steps "${TOTAL_STEPS}" \
  --batch-size "${BATCH_SIZE}" --grad-accum "${GRAD_ACCUM}" --max-seq-len "${MAX_SEQ_LEN}" \
  --save-every "${SAVE_EVERY}" --eval-every "${EVAL_EVERY}" --log-every "${LOG_EVERY}" \
  --keep-checkpoints "${KEEP_CKPTS}" --dtype "${DTYPE}" --seed "${SEED}" \
  --wandb-project "${WANDB_PROJECT}" \
  ${RESUME_ARG} \
  2>&1 | tee "${OUTPUT_DIR}/train_${TIMESTAMP}.log"

EXIT=${PIPESTATUS[0]}
[ ${EXIT} -eq 0 ] && echo "Done. Model: ${OUTPUT_DIR}/final/" || \
  echo "Failed (${EXIT}). Resume: ./scripts/train.sh --resume \$(ls -td ${OUTPUT_DIR}/step_* | head -1)"
exit ${EXIT}
