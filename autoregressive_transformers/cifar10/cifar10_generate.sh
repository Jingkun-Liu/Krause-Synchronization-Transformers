#!/bin/bash

GPU_ID="0"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D_MODEL=512
NUM_LAYERS=16
GENERATE_SAMPLES=10000
TOP_K=192
WINDOW_SIZE=256
GEN_BATCH_SIZE=200

SCRIPT_NAME="generate_cifar10.py"

echo "Starting image transformer image generation script..."
echo "Loading model parameters: D_MODEL=${D_MODEL}, L=${NUM_LAYERS}"
echo "Generating samples: ${GENERATE_SAMPLES}"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: '$SCRIPT_NAME' file not found. Please ensure the script is in the current directory."
    exit 1
fi

MODEL_DIR="saved_models_cifar10"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Warning: model directory '$MODEL_DIR' not found. Please put the trained model files in this directory."
fi


echo "Running $SCRIPT_NAME ..."

python3 "$SCRIPT_NAME" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --generate_samples "$GENERATE_SAMPLES" \
    --top_k "$TOP_K" \
    --window_size "$WINDOW_SIZE" \
    --gen_batch_size "$GEN_BATCH_SIZE"

if [ $? -eq 0 ]; then
    echo "Script executed successfully!"
    echo "Generated images should be saved to: generation_results_cifar10/generated_samples_cifar10.png"
else
    echo "Script execution failed. Please check the error information, model file path, and parameters."
fi
