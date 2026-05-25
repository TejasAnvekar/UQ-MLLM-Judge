#!/bin/bash

# Ensure Bash semantics even if launched via `sh scripts/run.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

# =====================================================
# Run score, pair, and batch evaluations sequentially
# with one shared vLLM server.
#
# Usage:
#   ./scripts/run_all.sh 7
#   ./scripts/run_all.sh 6,7
# =====================================================

if [ $# -ge 1 ] && [ -n "${1:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$1"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
TP_SIZE="$NUM_GPUS"

# =====================================================
# CONFIG (override via env vars)
# =====================================================

MODEL_NAME="${MODEL_NAME:-google/gemma-3-4b-it}"
GPU_MEM="${GPU_MEM:-0.85}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_API_BASE="${VLLM_API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"

LIMIT_MM_PER_PROMPT_IMAGE="${LIMIT_MM_PER_PROMPT_IMAGE:-1}"

PROJECT_DIR="${PROJECT_DIR:-/gscratch/tkishore/MLLM-Judge}"
SCRIPT_PATH="${SCRIPT_PATH:-/gscratch/tkishore/MLLM-Judge/scripts/uq_vllm_conformal_all.py}"

LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/Experiments/logs/UQ/}"
mkdir -p "$LOG_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"

OUT_DIR="${OUT_DIR:-${RUN_LOG_DIR}/Figures/UQ}"

CONCURRENCY="${CONCURRENCY:-128}"
CAL_FRAC="${CAL_FRAC:-0.3}"
SEED="${SEED:-42}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"

ALPHAS="${ALPHAS:-0.01,0.05,0.1,0.15,0.2}"

MAX_SAMPLES="${MAX_SAMPLES:-}"

TASK_ORDER="${TASK_ORDER:-score pair batch}"

DATASET_SCORE="${DATASET_SCORE:-${PROJECT_DIR}/Dataset/Benchmark_Lite/score_lite.jsonl}"
DATASET_PAIR="${DATASET_PAIR:-${PROJECT_DIR}/Dataset/Benchmark_Lite/pair_lite.jsonl}"
DATASET_BATCH="${DATASET_BATCH:-${PROJECT_DIR}/Dataset/Benchmark_Lite/batch_lite.jsonl}"

IMAGE_ROOT="${IMAGE_ROOT:-${PROJECT_DIR}/Dataset/image}"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "Starting vLLM + UQ multi-task run"
echo "Model: $MODEL_NAME"
echo "GPU ids: $CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size: $TP_SIZE"
echo "VLLM: $VLLM_API_BASE (api_key=$VLLM_API_KEY)"
echo "Task order: $TASK_ORDER"
echo "Score dataset: $DATASET_SCORE"
echo "Pair dataset:  $DATASET_PAIR"
echo "Batch dataset: $DATASET_BATCH"
echo "Image root: $IMAGE_ROOT"
echo "CONCURRENCY: $CONCURRENCY"
echo "Calibration fraction: $CAL_FRAC"
echo "Temperature: $TEMPERATURE / top_p: $TOP_P"
echo "top_logprobs: $TOP_LOGPROBS"
echo "Outputs: $OUT_DIR"
echo "Log dir: $RUN_LOG_DIR"
echo "=========================================="

check_server_ready() {
  local health_url="${VLLM_API_BASE%/v1}/health"
  for i in {1..60}; do
    if curl -s -f "$health_url" > /dev/null 2>&1; then
      echo "[OK] vLLM server ready"
      return 0
    fi
    echo "Waiting for server... ($i/60)"
    sleep 5
  done
  echo "[ERROR] Server failed to start"
  return 1
}

cleanup() {
  echo ""
  echo "Stopping vLLM..."
  if [ -n "${VLLM_PID:-}" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Launching vLLM..."
vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_MEM" \
  --api-key "$VLLM_API_KEY" \
  --chat-template-content-format openai \
  --limit-mm-per-prompt.image "$LIMIT_MM_PER_PROMPT_IMAGE" \
  > "${RUN_LOG_DIR}/vllm.log" 2>&1 &

VLLM_PID=$!

if ! check_server_ready; then
  echo "Check log: ${RUN_LOG_DIR}/vllm.log"
  exit 1
fi

run_task() {
  local task_type="$1"
  local dataset_path="$2"
  local task_log="${RUN_LOG_DIR}/${task_type}.log"

  if [ ! -f "$dataset_path" ]; then
    echo "[WARN] Skipping ${task_type}: dataset not found at ${dataset_path}"
    return 0
  fi

  echo ""
  echo "=========================================="
  echo "Running task: ${task_type}"
  echo "Dataset: ${dataset_path}"
  echo "Log: ${task_log}"
  echo "=========================================="

  UQ_ARGS=(
    "$SCRIPT_PATH"
    --vllm_model "$MODEL_NAME"
    --api_base_url "$VLLM_API_BASE"
    --api_key "$VLLM_API_KEY"
    --dataset_path "$dataset_path"
    --image_root "$IMAGE_ROOT"
    --out_dir "$OUT_DIR"
    --concurrency "$CONCURRENCY"
    --calibration_fraction "$CAL_FRAC"
    --seed "$SEED"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_logprobs "$TOP_LOGPROBS"
    --alphas "$ALPHAS"
    --task_type "$task_type"
  )

  if [ -n "$MAX_SAMPLES" ]; then
    UQ_ARGS+=( --max_samples "$MAX_SAMPLES" )
  fi

  echo "Running: python ${UQ_ARGS[*]}"
  python "${UQ_ARGS[@]}" > "$task_log" 2>&1

  echo "[OK] Finished ${task_type}"
}

for task in $TASK_ORDER; do
  case "$task" in
    score)
      run_task "score" "$DATASET_SCORE"
      ;;
    pair)
      run_task "pair" "$DATASET_PAIR"
      ;;
    batch)
      run_task "batch" "$DATASET_BATCH"
      ;;
    *)
      echo "[WARN] Unknown task in TASK_ORDER: $task"
      ;;
  esac
done

echo ""
echo "=========================================="
echo "All runs complete."
echo "Outputs under: $OUT_DIR"
echo "Logs under: $RUN_LOG_DIR"
echo "=========================================="
