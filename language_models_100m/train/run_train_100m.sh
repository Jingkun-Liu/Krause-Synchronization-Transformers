MASTER_LOG="${MASTER_LOG:-run_tfsc_master.log}"

if [[ "${1:-}" != "--inner" ]]; then
  cd "$(dirname "$0")" || exit 1
  nohup bash "$0" --inner >> "$MASTER_LOG" 2>&1 &
  echo "Started background PID $!, master log: $(pwd)/$MASTER_LOG"
  exit 0
fi
shift

set -euo pipefail
cd "$(dirname "$0")" || exit 1

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 --master_port 33243 train_100m.py \
    --data_dir ./datasets/fwe10bt \
    --tokenizer_path ./llm/gpt2 \
    --output_root ./models_100m \
    --device cuda \
    --compile \
    --warmup_ratio 0.05 \
    --init_sigma 2.5 \
    > "train_100m_2.5_krause.log" 2>&1

