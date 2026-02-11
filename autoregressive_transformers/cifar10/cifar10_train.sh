#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

SCRIPT_DIR="./"
SCRIPT_NAME="train_cifar10.py"

LOG_DIR="${SCRIPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.*}_ddp_gpus${NUM_GPUS}_${TIMESTAMP}.log"

NUM_GPUS=4
EPOCHS=30
BATCH_SIZE=2
D_MODEL=512
NUM_LAYERS=16
TOP_K=192
WINDOW_SIZE=256

echo "Training started. Output will be printed to the terminal and saved to: $LOG_FILE"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="29498" \
    "$SCRIPT_NAME" \
    --epochs "$EPOCHS" \
    --window_size "$WINDOW_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --top_k "$TOP_K" \
    > $LOG_FILE 2>&1 &

echo "Training started."
