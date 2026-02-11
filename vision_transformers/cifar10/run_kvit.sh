#!/bin/bash

SIGMAS="2.5"
DROPOUTS="0.0"

EPOCHS=300
LR=3e-4
WEIGHT_DECAY=0.05
BATCH_SIZE=256

echo "--- Starting Krause-ViT Experiments (No Iteration Version) ---"
echo "Sigmas: $SIGMAS"
echo "Dropouts: $DROPOUTS"
echo "----------------------------------------"

for SIGMA in $SIGMAS; do
    for DROPOUT in $DROPOUTS; do
        
        FILE_SUFFIX="topk2-4_s${SIGMA}_d${DROPOUT}_w${WEIGHT_DECAY}_batchsize${BATCH_SIZE}"
        
        LOG_FILE="log_kvit_s_main_${FILE_SUFFIX}.out"
        
        SAVE_PATH="analysis_kvit_s_main_${FILE_SUFFIX}.png"
        
        echo ""
        echo "Running: sigma=${SIGMA}, dropout=${DROPOUT}"
        echo "Log file: ${LOG_FILE}"
        echo "Save path: ${SAVE_PATH}"
        
        CUDA_VISIBLE_DEVICES=0 nohup python3 kvit_s_main.py \
            --top_k 2 \
            --warmup_epoch 10 \
            --sigma $SIGMA \
            --dropout $DROPOUT \
            --epochs $EPOCHS \
            --lr $LR \
            --weight_decay $WEIGHT_DECAY \
            --batch_size $BATCH_SIZE \
            --save_path $SAVE_PATH \
            > $LOG_FILE 2>&1 &
        
        sleep 2
    done
done


echo ""
echo "--- All experiments launched. ---"
