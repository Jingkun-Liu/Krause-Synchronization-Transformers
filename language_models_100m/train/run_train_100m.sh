env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 --master_port 33243 train_100m.py \
    --data_dir /yourpath/datasets/fwe10bt \
    --tokenizer_path /yourpath/llm/gpt2 \
    --output_root /yourpath/models_100m \
    --device cuda \
    --compile \
    --warmup_ratio 0.05 \
    --init_sigma 2.5 \
    > "train_100m_2.5_krause.log" 2>&1

