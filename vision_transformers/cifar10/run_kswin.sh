#!/bin/bash

SIGMAS="4.0"

EPOCHS=300
LR=2e-3
WEIGHT_DECAY=0.01
BATCH_SIZE=256

echo "--- Starting Krause-Swin Experiments (No Iteration Version) ---"
echo "Sigmas: $SIGMAS"
echo "----------------------------------------"

for SIGMA in $SIGMAS; do

    FILE_SUFFIX="s${SIGMA}_w${WEIGHT_DECAY}_batchsize${BATCH_SIZE}"

    LOG_FILE="log_kswin_t_main_${FILE_SUFFIX}.out"

    SAVE_PATH="analysis_kswin_t_main_${FILE_SUFFIX}.png"

    echo ""
    echo "Running: sigma=${SIGMA}"
    echo "Log file: ${LOG_FILE}"
    echo "Save path: ${SAVE_PATH}"
    
    CUDA_VISIBLE_DEVICES=7 nohup python3 kswin_t_main.py \
        --warmup_epoch 20 \
        --sigma $SIGMA \
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
