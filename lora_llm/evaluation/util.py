import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

GLOBAL_LOCAL_RANK = 0
GLOBAL_IS_MAIN_PROCESS = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_rank0(*args, **kwargs):
    if GLOBAL_IS_MAIN_PROCESS:
        print(*args, **kwargs)

def init_distributed():
    global GLOBAL_LOCAL_RANK, GLOBAL_IS_MAIN_PROCESS

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        GLOBAL_LOCAL_RANK = local_rank
        GLOBAL_IS_MAIN_PROCESS = (rank == 0)

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        print_rank0(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True
    return False

def sync_distributed_results(correct, total):
    if dist.is_initialized():
        tensor_correct = torch.tensor([correct], device=torch.cuda.current_device())
        tensor_total = torch.tensor([total], device=torch.cuda.current_device())
        dist.all_reduce(tensor_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_total, op=dist.ReduceOp.SUM)
        return tensor_correct.item(), tensor_total.item()
    return correct, total

def load_model_and_tokenizer(model_path: str, base_model_path: str = None, device: str = "cuda", use_ddp: bool = False):
    print_rank0(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_ddp and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print_rank0(f"Barrier before model load failed: {e}")
            raise
        print_rank0(f"Process {dist.get_rank()} loading model to device {device}")
        torch.cuda.set_device(GLOBAL_LOCAL_RANK)

    model = _load_model_internal(model_path, base_model_path, device)
    
    model.eval()
    
    if use_ddp and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print_rank0(f"Barrier after model load failed: {e}")
            raise

    print_rank0("Model loaded")
    
    return model, tokenizer

def _load_model_internal(model_path: str, base_model_path: str = None, device="cuda"):
    use_ddp = dist.is_initialized()
    if use_ddp:
        target_device_index = GLOBAL_LOCAL_RANK
    else:
        if isinstance(device, str) and ":" in device:
            target_device_index = int(device.split(":")[-1])
        else:
            target_device_index = 0

    torch.cuda.set_device(target_device_index)

    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": {"": target_device_index} 
    }

    if os.path.exists(adapter_config_path):
        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0("Loading PEFT adapter...")

        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", None)

        actual_base_model_path = base_model_path if base_model_path else base_model_name

        if not actual_base_model_path:
            raise ValueError("Base model path could not be determined. Please specify via --base_model.")

        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f"Loading base model: {actual_base_model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            actual_base_model_path,
            **load_kwargs
        )
        
        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f"Loading PEFT adapter and merging: {model_path}")

        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f"Loading base model (no adapter): {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )

    return model
