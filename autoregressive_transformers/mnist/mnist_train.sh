#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5,6,7"

SCRIPT_DIR="./"
SCRIPT_NAME="train_mnist.py"

LOG_DIR="${SCRIPT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME%.*}_${TIMESTAMP}.log"

NUM_GPUS=4

EPOCHS=30
BATCH_SIZE=16
D_MODEL=256
NUM_LAYERS=12
TOP_K=96
WINDOW_SIZE=128

echo "Switch to script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR"
echo "DDP training started. Output will be printed to the terminal and saved to the file: $LOG_FILE"
echo "Using GPU IDs: $CUDA_VISIBLE_DEVICES"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="27777" \
    "$SCRIPT_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --top_k "$TOP_K" \
    --window_size "$WINDOW_SIZE" \
    > $LOG_FILE 2>&1 &

echo "DDP training command has been executed."
