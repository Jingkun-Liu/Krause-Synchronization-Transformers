# Krause Synchronization Transformers
This repository contains the implementation for the paper Krause Synchronization Transformers. In our work, we introduce <strong>Krause Attention</strong>, a principled attention mechanism inspired by bounded-confidence consensus dynamics. Krause Attention replaces similarity-based global aggregation with distance-based, localized, and selectively sparse interactions, promoting structured local synchronization instead of global mixing. We relate this behavior to recent theory modeling Transformer dynamics as interacting particle systems, and show how bounded-confidence interactions naturally moderate attention concentration and alleviate attention sinks. Restricting interactions to local neighborhoods also reduces runtime complexity from quadratic to linear in sequence length. Experiments across vision (ViT on CIFAR/ImageNet), autoregressive generation (MNIST/CIFAR-10), and large language models (Llama/Qwen) demonstrate consistent gains with substantially reduced computation, highlighting bounded-confidence dynamics as a scalable and effective inductive bias for attention.

<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <div class="has-text-centered">
        <img src="images/figure1.png" 
             alt="Description of the new attention mechanism" 
             style="width: 100%; height: auto; display: inline-block;"> 
             </div>
    </div>
  </div>
</section>

<section class="hero is-small">
  <div class="hero-body" style="background: transparent;">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-left">Krause Attention</h2>
        <div class="has-text-centered">
          <img src="images/kst_gif.gif" alt="Teaser GIF" style="width: 100%; height: auto; display: block;">
        </h2>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small">
  <div class="hero-body" style="background-color: #f5f5f5 !important; padding: 40px 0;">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-left">Alleviating Attention Sinks in Krause-LLMs</h2>
        <div class="has-text-centered">
          <img src="images/attention_sink_llama3.png" alt="Second research result visualization" loading="lazy" style="max-width: 100%; height: auto; display: block;"/>
        </div>
        </h2>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small" style="background: transparent;">
  <div class="hero-body">
    <div class="container">
      <div class="item" style="max-width: 850px; margin: 0 auto;">        
        <h2 class="title is-2.5 has-text-left">Attention Heatmaps in Vision Transformers</h2>        
        <div class="has-text-centered">
          <img src="images/imagenet_heatmap_main.png" 
               alt="Attention Heatmaps" 
               loading="lazy" 
               style="max-width: 100%; height: auto; display: block;"/>
        </div>
        <div class="has-text-centered">
          <img src="images/attention_evolution_map.png" 
               alt="Attention Evolution" 
               loading="lazy" 
               style="max-width: 100%; height: auto; display: block;"/>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="hero is-small" style="background-color: #f5f5f5 !important; padding: 40px 0;">
  <div class="hero-body">
    <div class="container">
      <div class="item" style="max-width: 1000px; margin: 0 auto;">
        <h2 class="title is-2.5 has-text-centered">Krause Autoregressive Transformers for Image Generation</h2>        
        <div class="columns is-vcentered is-variable is-5">
          <div class="column">
            <div class="has-text-centered">
              <img src="images/completion_mnist.png" 
                   alt="Attention Heatmaps" 
                   loading="lazy" 
                   style="width: 100%; height: auto; display: block;"/>
            </div>
          </div>
          <div class="column">
            <div class="has-text-centered">
              <img src="images/completion_cifar10.png" 
                   alt="Attention Evolution Across Layers" 
                   loading="lazy" 
                   style="width: 100%; height: auto; display: block;"/>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

## Installation

To get started with Krause-Synchronization-Transformers, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Jingkun-Liu/Krause-Synchronization-Transformers.git](https://github.com/Jingkun-Liu/Krause-Synchronization-Transformers.git)
cd Krause-Synchronization-Transformers
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

## Project Structure
```text
Krause-Synchronization-Transformers/
├── autoregressive_transformers/
│   ├── cifar10/ 
│   │   │── cifar10_generate.sh
│   │   │── cifar10_train.sh
│   │   │── completion_cifar10.py
│   │   │── generate_cifar10.py
│   │   └── train_cifar10.py
│   ├── mnist/
│   │   │── mnist_generate.sh
│   │   │── mnist_train.sh
│   │   │── completion_mnist.py
│   │   │── generate_mnist.py
│   │   └── train_mnist.py
├── vision_transformers/
│   ├── cifar10/ 
│   │   │── ViT-S/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── vit_s_main.py
│   │   │── KViT-S/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── kvit_s_main.py
│   │   │── Swin-T/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── swin_t_main.py
│   │   │── KSwin-T/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── kswin_t_main.py
│   │   │── run_kswin.sh
│   │   │── run_kvit.sh
│   │   │── run_swin.sh
│   │   │── run_vit.sh
│   ├── imagenet1k/ 
│   │   │── KViT-S-16/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── kvit_s_16_main.py
│   │   │── ViT-S-16/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── vit_s_16_main.py
│   │   │── KViT-B-16/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── kvit_b_16_main.py
│   │   │── ViT-B-16/
│   │   │   │── data.py
│   │   │   │── module.py
│   │   │   └── vit_b_16_main.py
│   │   │── run_kvit.sh
│   │   │── run_vit.sh
├── lora_llms/
│   ├── llama/ 
│   │   │── module.py
│   │   │── util.py
│   │   │── run_llama3_8b.sh
│   │   └── llama3_8b_main.py
│   ├── qwen/ 
│   │   │── module.py
│   │   │── util.py
│   │   │── run_qwen1.5_7b.sh
│   │   └── qwen1.5_7b_main.py
│   └── evaluation/ 
│       │── benchmark.py
│       │── util.py
│       │── evaluation.sh
│       └── main.py             
└── images/  # images/gifs used in readme and our website
```

## Datasets
* **Automatic Download**: The `CIFAR-10` and `MNIST` datasets will be automatically downloaded upon running the scripts.
* **Manual Download Required**:
    * **ImageNet-1K**: Please download from [https://www.image-net.org/download.php].
    * **LLM Datasets**: Relevant datasets can be found at [https://huggingface.co/datasets/SirNeural/flan_v2/tree/main].
    * **LLMs**: Llama3-8B can be found at [https://huggingface.co/meta-llama/Meta-Llama-3-8B]. Qwen1.5-7B can be found at [https://huggingface.co/Qwen/Qwen1.5-7B].
---
> **Local Dataset Release**
> We have also prepared a set of locally curated datasets optimized for this project, which will be released soon to ensure reproducibility.

## Model Checkpoints
ckpts will be released soon

## Usage
We provide run scripts that can be submitted simply using sbatch for every task. For example, to run the ImageNet-1K classification task for KViT-S-16, use the following command:
```bash
/Krause-Synchronization-Transformers-main/vision_transformers/imagenet1k/run_kvit.sh
```
> [!IMPORTANT]
> **Script Customization:**
> Although we provide templates for various models, they are not designed for every specific parameter scale. **Please ensure you modify the script's configuration** (such as batch size, learning rate, model implementation path or GPU requirements) before execution.
>
> For instance, to run **ImageNet-1K** with **KViT-S-16**, the script should be adjusted as shown below:
```bash
# Example script
#!/bin/bash

SIGMAS="4.5"
DROPOUTS="0.0"

EPOCHS=300
LR=5e-4
WEIGHT_DECAY=0.05
BATCH_SIZE=512
NPROC_PER_NODE=2

echo "=========================================="
echo "Starting Krause-ViT 4-GPU DDP Experiment"
echo "=========================================="
echo "Sigmas: $SIGMAS"
echo "Dropouts: $DROPOUTS"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NPROC_PER_NODE))"
echo "=========================================="

for SIGMA in $SIGMAS; do
    for DROPOUT in $DROPOUTS; do
      
        echo ""
        echo "=========================================="
        echo "New Training Run"
        echo "=========================================="
        
        FILE_SUFFIX="topk8-16_s${SIGMA}_d${DROPOUT}_w${WEIGHT_DECAY}_batchsize${BATCH_SIZE}"
      
        LOG_FILE="log_kvits16_ImageNet_lr5e-4_${FILE_SUFFIX}.out"

        SAVE_PATH="analysis_kvits16_ImageNet_lr5e-4_${FILE_SUFFIX}.png"

        echo ""
        echo "Task configuration:"
        echo "  - Sigma: ${SIGMA}"
        echo "  - Dropout: ${DROPOUT}"
        echo "  - Master Port: ${FREE_PORT}"
        echo "  - Log file: ${LOG_FILE}"
        echo "  - Save path: ${SAVE_PATH}"
        echo "  - Batch size per GPU: $BATCH_SIZE"
        echo "  - Total batch size: $((BATCH_SIZE * NPROC_PER_NODE))"
        echo ""
        
        echo "Starting training..."
        CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=27902 kvit_b_16_main.py \
            --top_k 8 \
            --warmup_epochs 10 \
            --sigma $SIGMA \
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
```

## Citation
If you find this research useful, please consider citing our work!
```bash
@article{liukrause2026,
  title={Krause Synchronization Transformers},
  author={Jingkun Liu and Yisong Yue and Max Welling and Yue Song},
  journal={ArXiv},
  year={2026},
  url={https://your-domain.com/your-project-page}
}
```
