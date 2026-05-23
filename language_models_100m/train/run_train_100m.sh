MASTER_LOG="${MASTER_LOG:-run_tfsc_revision_master.log}"

if [[ "${1:-}" != "--inner" ]]; then
  cd "$(dirname "$0")" || exit 1
  nohup bash "$0" --inner >> "$MASTER_LOG" 2>&1 &
  echo "Started background PID $!, master log: $(pwd)/$MASTER_LOG"
  exit 0
fi
shift

set -euo pipefail
cd "$(dirname "$0")" || exit 1

WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SIGMAS=(2.5)

for SIGMA in "${SIGMAS[@]}"; do
  echo "========== init_sigma=${SIGMA} WARMUP_RATIO=${WARMUP_RATIO} $(date -Is) =========="
  env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 --master_port 33243 train_100m.py \
      --data_dir ./datasets/fwe10bt \
      --tokenizer_path ./llm/gpt2 \
      --output_root ./models_100m \
      --device cuda \
      --compile \
      --warmup_ratio "${WARMUP_RATIO}" \
      --init_sigma "${SIGMA}" \
      > "train_100m_${SIGMA}_krause.log" 2>&1
done

echo "All sigma runs finished. $(date -Is)"
