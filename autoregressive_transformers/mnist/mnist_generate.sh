#!/bin/bash

GPU_ID="0"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D_MODEL=256
NUM_LAYERS=12
GENERATE_SAMPLES=50000
TOP_K=96
WINDOW_SIZE=128
GEN_BATCH_SIZE=2000

SCRIPT_NAME="generate_mnist.py"

echo "Starting image transformer image generation script..."
echo "Loading model parameters: D_MODEL=${D_MODEL}, L=${NUM_LAYERS}"
echo "Generating samples: ${GENERATE_SAMPLES}"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: '$SCRIPT_NAME' file not found. Please ensure the script is in the current directory."
    exit 1
fi

MODEL_DIR="saved_models_mnist"
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
    echo "Script executed successfully"
    echo "Generated result images should be saved to: generation_results_mnist/generated_samples.png"
else
    echo "Script executed failed. Please check the error information or model file path."
fi
