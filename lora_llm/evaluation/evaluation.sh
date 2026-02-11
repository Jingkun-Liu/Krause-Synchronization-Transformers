#!/bin/bash

GPUS="0,1,2,3"

SCRIPT_NAME="main.py"
NUM_GPUS=4
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S)_krause.log"

echo "----------------------------------------------------------------"
echo "Starting distributed evaluation"
echo "Using GPUs: $GPUS ($NUM_GPUS GPUs)"
echo "Log file: $LOG_FILE"
echo "----------------------------------------------------------------"

export CUDA_VISIBLE_DEVICES=$GPUS
export NCCL_DEBUG=WARN

export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NET_GDR_LEVEL=0
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port="38245" \
    "$SCRIPT_NAME" > "$LOG_FILE" 2>&1 &
