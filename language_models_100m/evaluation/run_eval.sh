set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLI_MODEL_PATH=""
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      if [[ $# -lt 2 ]]; then
        exit 1
      fi
      CLI_MODEL_PATH="$2"
      shift 2
      ;;
    --model_path=*)
      CLI_MODEL_PATH="${1#*=}"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${PASSTHROUGH_ARGS[@]}"

DEFAULT_MODEL_PATH="${SCRIPT_DIR}/models_100m/krause_sigma_2.5/hf_model-krause"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
if [[ -n "${CLI_MODEL_PATH}" ]]; then
  MODEL_PATH="${CLI_MODEL_PATH}"
fi

TOKENIZER_NAME="${TOKENIZER_NAME:-${SCRIPT_DIR}/llm/gpt2}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}}"
TASKS="${TASKS:-hellaswag,piqa,blimp,arc_e,cbt,lambada}"
ARC_E_SPLIT="${ARC_E_SPLIT:-test}"
CBT_SPLIT="${CBT_SPLIT:-test}"
LAMBADA_JSONL="${LAMBADA_JSONL:-datasets/lambada/data/lambada_test_en.jsonl}"
HELLASWAG_SPLIT="${HELLASWAG_SPLIT:-validation}"
PIQA_SPLIT="${PIQA_SPLIT:-validation}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
OUTPUT_JSON="${OUTPUT_JSON:-eval_results.json}"

EVAL_DEFAULT_ARGS=(
  --model_path "${MODEL_PATH}"
  --tokenizer_name "${TOKENIZER_NAME}"
  --data_root "${DATA_ROOT}"
  --tasks "${TASKS}"
  --arc_e_split "${ARC_E_SPLIT}"
  --cbt_split "${CBT_SPLIT}"
  --lambada_jsonl "${LAMBADA_JSONL}"
  --hellaswag_split "${HELLASWAG_SPLIT}"
  --piqa_split "${PIQA_SPLIT}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --output_json "${OUTPUT_JSON}"
)

GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi
_count_gpus() {
  awk -F',' 'NF>0 {print NF; exit}' <<< "${1}"
}

_auto_nproc="$(_count_gpus "${CUDA_VISIBLE_DEVICES}")"
if [[ -z "${_auto_nproc}" || "${_auto_nproc}" -lt 1 ]]; then
  _auto_nproc=1
fi

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  :
elif [[ -n "${NPROC:-}" ]]; then
  NPROC_PER_NODE="${NPROC}"
else
  NPROC_PER_NODE="${_auto_nproc}"
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi

MASTER_PORT="${MASTER_PORT:-38942}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_eval}"
if [[ -z "${LOG_FILE:-}" ]]; then
  LOG_FILE="${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/eval.py" \
    "${EVAL_DEFAULT_ARGS[@]}" \
    "$@"
fi

nohup torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/eval.py" \
  "${EVAL_DEFAULT_ARGS[@]}" \
  "$@" >> "$LOG_FILE" 2>&1 &
