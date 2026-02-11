#!/bin/bash

GPUS="0,1,2,3"
SCRIPT_NAME="llama3_8b_main.py"
TRAIN_MODE=${1:-krause}
NUM_GPUS=4
LOG_FILE="${TRAIN_MODE}_training_$(date +%Y%m%d_%H%M%S)_llama3_8b.log"

echo "----------------------------------------------------------------"
echo "Starting distributed training"
echo "Running mode: $TRAIN_MODE"
echo "Using GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "Log file path: $LOG_FILE"
echo "----------------------------------------------------------------"

export CUDA_VISIBLE_DEVICES=$GPUS
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRAIN_MODE=$TRAIN_MODE

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port="37777" \
    "$SCRIPT_NAME" \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "Training has been started in the background"
echo "Background process ID (PID): $PID"
