#!/bin/bash

DROPOUTS="0.0"

EPOCHS=300
LR=3e-4
WEIGHT_DECAY=0.05
BATCH_SIZE=256

echo "--- Starting ViT Experiments---"
echo "Dropouts: $DROPOUTS"
echo "----------------------------------------"

for DROPOUT in $DROPOUTS; do
    
    FILE_SUFFIX="d${DROPOUT}_w${WEIGHT_DECAY}_batchsize${BATCH_SIZE}"
    
    LOG_FILE="log_vit_s_main_${FILE_SUFFIX}.out"
    
    SAVE_PATH="analysis_vit_s_main_${FILE_SUFFIX}.png"
    
    echo ""
    echo "Running: dropout=${DROPOUT}"
    echo "Log file: ${LOG_FILE}"
    echo "Save path: ${SAVE_PATH}"
    
    CUDA_VISIBLE_DEVICES=0 nohup python3 vit_s_main.py \
        --warmup_epoch 10 \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --batch_size $BATCH_SIZE \
        --save_path $SAVE_PATH \
        > $LOG_FILE 2>&1 &
    
    sleep 2
done



echo ""
echo "--- All experiments launched. ---"
