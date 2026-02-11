#!/bin/bash

DROPOUTS="0.0"

EPOCHS=300
LR=1e-3
WEIGHT_DECAY=0.05
BATCH_SIZE=256
NPROC_PER_NODE=4

echo "=========================================="
echo "Starting ViT-B-16 4-GPU DDP Experiment"
echo "=========================================="
echo "Dropouts: $DROPOUTS"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NPROC_PER_NODE))"
echo "=========================================="

for DROPOUT in $DROPOUTS; do
    
    echo ""
    echo "=========================================="
    echo "New Training Run"
    echo "=========================================="
    
    FILE_SUFFIX="d${DROPOUT}_w${WEIGHT_DECAY}_batchsize${BATCH_SIZE}"
    
    LOG_FILE="log_vitb16_ImageNet_lr1e-3_${FILE_SUFFIX}.out"

    SAVE_PATH="analysis_vitb16_ImageNet_lr1e-3_${FILE_SUFFIX}.png"

    echo ""
    echo "Task configuration:"
    echo "  - Dropout: ${DROPOUT}"
    echo "  - Master Port: ${FREE_PORT}"
    echo "  - Log file: ${LOG_FILE}"
    echo "  - Save path: ${SAVE_PATH}"
    echo "  - Batch size per GPU: $BATCH_SIZE"
    echo "  - Total batch size: $((BATCH_SIZE * NPROC_PER_NODE))"
    echo ""
    
    echo "Starting training..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=27922 vit_b_16_main.py \
        --warmup_epochs 10 \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --batch_size $BATCH_SIZE \
        --save_path $SAVE_PATH \
        > $LOG_FILE 2>&1 &
    
    TRAIN_PID=$!
    echo "Training started (PID: $TRAIN_PID, Port: $FREE_PORT)"
    
    echo "$TRAIN_PID $FREE_PORT $LOG_FILE $SAVE_PATH" >> training_processes.txt
    
    sleep 5
    
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "Training process running normally"
    else
        echo "Warning: Training process may have failed, please check the log: $LOG_FILE"
    fi
    
    echo "=========================================="
done



echo ""
echo "=========================================="
echo "All training tasks have been started"
echo "=========================================="
echo ""
echo "Monitoring commands:"
echo "  - Check log: tail -f ${LOG_FILE}"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - Check processes: cat training_processes.txt"
echo ""
echo "Stop training:"
echo "  - Stop all: pkill -f ddp_torch.py"
echo "  - Stop specific: kill <PID>"
echo ""
echo "Training process information saved to: training_processes.txt"
echo "=========================================="
