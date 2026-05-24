export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

exec torchrun \
  --nproc_per_node=4 \
  --master_port=38942 \
  ./eval.py \
  --model_path /yourpath/models_100m/krause_sigma_2.5/hf_model-krause \
  --tokenizer_name /yourpath/gpt2 \
  --data_root /yourpath \
  --tasks hellaswag,piqa,blimp,arc_e,cbt,lambada \
  --arc_e_split test \
  --cbt_split test \
  --lambada_jsonl /yourpath/datasets/lambada/data/lambada_test_en.jsonl \
  --hellaswag_split validation \
  --piqa_split validation \
  --max_seq_length 1024 \
  --output_json eval_results.json \
