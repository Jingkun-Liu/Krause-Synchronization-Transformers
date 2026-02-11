import os
import sys
import glob
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from util import (
    set_seed,
    print_rank0,
    ResponseOnlyDataCollator,
    run_sink_analysis,
    replace_attention_modules,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MAX_SEQ_LENGTH = int(os.environ.get('MAX_SEQ_LENGTH', '2048'))

set_seed(3407)

GLOBAL_LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
GLOBAL_IS_MAIN_PROCESS = (GLOBAL_LOCAL_RANK == 0)


def train_and_benchmark(mode='krause', model_path="Llama/Llama3-8B", flanv2_path=None):
    KRAUSE_PARAMS = {
        'top_k': 16, 
        'window_size':32, 
        'init_sigma': 5.5
    }

    if mode == 'krause':
        lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "q_proj_krause", "k_proj_krause", "v_proj_krause", "o_proj_krause",
            "gate_proj", "up_proj", "down_proj"
        ],
        modules_to_save=["gate_proj_krause"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
        )

    elif mode == 'baseline':
        lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        task_type="CAUSAL_LM"
        )

    training_args = TrainingArguments(
        output_dir=f"./output_{mode}_llama3_8b",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        learning_rate=5e-5,
        bf16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,  
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=None,
        report_to="none",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        ddp_timeout=7200,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=4,
        group_by_length=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    print_rank0(f"Loading model: {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": GLOBAL_LOCAL_RANK},
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    if hasattr(model, 'config'):
        model.config.use_cache = False
        print_rank0("use_cache=False (compatible with gradient checkpointing)")

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print_rank0("Gradient checkpointing enabled")

        
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    if mode != 'baseline':
        replace_attention_modules(model, mode, KRAUSE_PARAMS)
    
    model = get_peft_model(model, lora_config)
    
    if GLOBAL_IS_MAIN_PROCESS:
        model.print_trainable_parameters()
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print_rank0("Gradient checkpointing enabled")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    
    cache_dir = "./flanv2_processed_llama"
    train_cache_path = os.path.join(cache_dir, "train_5w")
    eval_cache_path = os.path.join(cache_dir, "eval_5k")

    if os.path.exists(train_cache_path):
        print_rank0(">>> Loading cache...")
        train_ds = load_from_disk(train_cache_path)
        if os.path.exists(eval_cache_path):
            eval_ds = load_from_disk(eval_cache_path)
        else:
            print_rank0(">>> Splitting evaluation set from training set...")
            eval_size = 5000
            eval_ds = train_ds.select(range(eval_size))
            train_ds = train_ds.select(range(eval_size, len(train_ds)))
           
    else:
        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f">>> Loading Flan-V2 data from local path: {flanv2_path}")
            
            all_json_files = glob.glob(os.path.join(flanv2_path, "*.json"))
            all_json_files = [f for f in all_json_files if not os.path.basename(f) in ['dataset_infos.json', 'README.md']]
            
            if not all_json_files:
                raise FileNotFoundError(
                    f"No JSON data files found in {flanv2_path}\n"
                    f"Please ensure the directory contains Flan-V2 JSON files"
                )

            train_ds = load_dataset(
                "json",
                data_files=all_json_files,
                split="train",
                num_proc=8
            )
            print_rank0(f">>> Successfully loaded Flan-V2 dataset, original size: {len(train_ds)}")
            
            def format_flan_item(example):
                instruction = example.get('inputs', '').strip()
                response = example.get('targets', '').strip()
                
                eos = tokenizer.eos_token 
                
                if instruction and response:
                    text = f"{instruction}\n\n### Response:\n{response}{eos}"
                elif response:
                    text = f"{response}{eos}"
                else:
                    text = ""
                
                return {'text': text}
            
            train_ds = train_ds.map(
                format_flan_item,
                num_proc=8,
                remove_columns=[col for col in train_ds.column_names if col != 'text'],
                desc="Formatting Flan-V2 data"
            )
            
            print_rank0(">>> Filtering empty text...")
            train_ds = train_ds.filter(
                lambda x: x['text'] is not None and len(x['text'].strip()) > 0, 
                num_proc=8
            )
            
            print_rank0(f">>> Formatted data size: {len(train_ds)}")

            target_total_size = 55000
            if len(train_ds) > target_total_size:
                train_ds = train_ds.shuffle(seed=3407)
                train_ds = train_ds.select(range(target_total_size))
            
            print_rank0(f">>> Final data: {len(train_ds)}")

            eval_size = 5000
            train_size = 50000
            eval_ds = train_ds.select(range(eval_size))
            train_ds = train_ds.select(range(eval_size, eval_size + train_size))

            os.makedirs(cache_dir, exist_ok=True)
            train_ds.save_to_disk(train_cache_path)
            eval_ds.save_to_disk(eval_cache_path)
            print_rank0(f">>> Saved {len(train_ds)} training data and {len(eval_ds)} evaluation data to disk")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        train_ds = load_from_disk(train_cache_path)
        eval_ds = load_from_disk(eval_cache_path)

    print_rank0(f"Data ready: {len(train_ds)} training samples, {len(eval_ds)} evaluation samples")
    
    eval_ds_for_sink = eval_ds
    
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    response_template = "### Response:"
    data_collator = ResponseOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if GLOBAL_IS_MAIN_PROCESS:
        print(f"Starting training mode: {mode.upper()}")
    
    trainer.train()
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier(timeout=300)

    
    if GLOBAL_IS_MAIN_PROCESS:
        model.save_pretrained(f"./final_model_{mode}_llama3_8b")
        tokenizer.save_pretrained(f"./final_model_{mode}_llama3_8b")
        print_rank0(f"Model saved to ./final_model_{mode}_llama3_8b")
    
    print_rank0("\nStarting Attention Sink analysis...")
    if GLOBAL_IS_MAIN_PROCESS:
        device = next(model.parameters()).device
        run_sink_analysis(model, tokenizer, device, eval_ds_for_sink, mode=mode, max_seq_length=MAX_SEQ_LENGTH)
    
    if torch.distributed.is_initialized():
        print_rank0("\nCleaning up distributed process group...")
        torch.distributed.barrier(timeout=300)
        torch.distributed.destroy_process_group()
    
    if GLOBAL_IS_MAIN_PROCESS:
        print(f"\nTraining and analysis completed (Mode: {mode.upper()})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Krause Llama model')
    parser.add_argument('--mode', type=str, default=None,
                       help='Training mode: krause or baseline')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the model')
    parser.add_argument('--flanv2_path', type=str, default=None,
                       help='Path to Flan-V2 dataset directory')
    
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    mode_from_env = args.mode if args.mode is not None else os.environ.get('TRAIN_MODE', 'krause')
    FLANV2_PATH = args.flanv2_path
    
    print_rank0(f"Training mode: {mode_from_env}, model path: {MODEL_PATH}")
    print_rank0(f"Flan-V2 dataset path: {FLANV2_PATH}")

    train_and_benchmark(mode=mode_from_env, model_path=MODEL_PATH, flanv2_path=FLANV2_PATH)
