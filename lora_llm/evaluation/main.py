import os
import json
import traceback
import argparse
import torch
import torch.distributed as dist

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_SHM_DISABLE'] = '0'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['NCCL_NET_GDR_LEVEL'] = '0'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from util import (
    set_seed,
    print_rank0,
    init_distributed,
    load_model_and_tokenizer,
    GLOBAL_LOCAL_RANK,
    GLOBAL_IS_MAIN_PROCESS,
)
from benchmark import (
    evaluate_boolq,
    evaluate_cb,
    evaluate_anli_r1,
    evaluate_anli_r2,
    evaluate_anli_r3,
    evaluate_piqa,
    evaluate_mnli,
    evaluate_mmlupro,
)

set_seed(3407)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LLM (DDP supported)')
    parser.add_argument('--model_path', type=str, default='./final_model_krause_qwen1.5_7b',
                       help='Path to trained model (e.g. final_model_*)')
    parser.add_argument('--base_model', type=str, default=None,
                       help='Base model path (for PEFT when auto-detect fails)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output results path (default: eval_results_<model_name>.json)')
    parser.add_argument('--tasks', type=str, nargs='+',
                       default=['boolq', 'cb', 'anli_r1', 'anli_r2', 'anli_r3', 'piqa', 'mnli', 'mmlupro'],
                       help='Tasks to run')

    args = parser.parse_args()

    use_ddp = init_distributed()

    if use_ddp:
        device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
    else:
        device = torch.device(args.device)
    
    device_arg = device if use_ddp else str(device)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, device_arg, use_ddp)
    
    eval_results = {}

    task_functions = {
        'boolq': evaluate_boolq,
        'cb': evaluate_cb,
        'anli_r1': evaluate_anli_r1,
        'anli_r2': evaluate_anli_r2,
        'anli_r3': evaluate_anli_r3,
        'piqa': evaluate_piqa,
        'mnli': evaluate_mnli,
        'mmlupro': evaluate_mmlupro
    }

    if use_ddp and dist.is_initialized():
        dist.barrier()
    
    for task in args.tasks:
        if task not in task_functions:
            if GLOBAL_IS_MAIN_PROCESS:
                print_rank0(f"Unknown task {task}, skipping")
            continue

        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f"\n{'='*60}")
            print_rank0(f"Task: {task.upper()}")
            print_rank0(f"{'='*60}")

        try:
            result = task_functions[task](model, tokenizer, device)
            
            if GLOBAL_IS_MAIN_PROCESS:
                eval_results[task] = result
                if "error" not in result:
                    print_rank0(f"{task.upper()} done: {result}")
                else:
                    print_rank0(f"{task.upper()} failed: {result['error']}")
        except KeyboardInterrupt:
            if GLOBAL_IS_MAIN_PROCESS:
                print_rank0(f"Evaluation {task} interrupted by user")
                eval_results[task] = {"error": "User interrupted"}
            raise
        except Exception as e:
            if GLOBAL_IS_MAIN_PROCESS:
                print_rank0(f"Error evaluating {task}: {e}")
                traceback.print_exc()
                eval_results[task] = {"error": str(e)}

        if use_ddp and dist.is_initialized():
            dist.barrier()
    
    if use_ddp and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print_rank0(f"Barrier failed: {e}")
    
    if GLOBAL_IS_MAIN_PROCESS:
        if args.output_file is None:
            model_name = os.path.basename(args.model_path)
            args.output_file = f"./eval_results_{model_name}.json"
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print_rank0(f"\nResults saved to: {args.output_file}")

        print_rank0("\n" + "="*80)
        print_rank0("Evaluation summary:")
        print_rank0("="*80)

        task_names = {
            'boolq': 'BoolQ',
            'cb': 'CB (CommitmentBank)',
            'anli_r1': 'ANLI R1',
            'anli_r2': 'ANLI R2',
            'anli_r3': 'ANLI R3',
            'piqa': 'PIQA',
            'mnli': 'MNLI',
            'mmlupro': 'MMLU-Pro'
        }
        
        all_tasks = ['boolq', 'cb', 'anli_r1', 'anli_r2', 'anli_r3', 'piqa', 'mnli', 'mmlupro']
        
        for task in all_tasks:
            if task not in eval_results:
                continue
                
            result = eval_results[task]
            task_display_name = task_names.get(task, task.upper())
            
            if "error" not in result:
                if task in ['cb', 'anli_r1', 'anli_r2', 'anli_r3', 'mnli']:
                    acc = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    correct = result.get('correct', 0)
                    total = result.get('total', 0)
                    print_rank0(f"{task_display_name:25s}: Accuracy = {acc:.4f}, F1 = {f1:.4f} ({correct}/{total})")
                else:
                    acc = result.get('accuracy', 0)
                    correct = result.get('correct', 0)
                    total = result.get('total', 0)
                    print_rank0(f"{task_display_name:25s}: Accuracy = {acc:.4f} ({correct}/{total})")
            else:
                print_rank0(f"{task_display_name:25s}: Failed - {result['error']}")

        print_rank0("="*80)

        valid_tasks = [task for task in all_tasks
                      if task in eval_results and 'error' not in eval_results[task]]

        if valid_tasks:
            print_rank0("\nAverage score:")
            accuracies = []
            for task in valid_tasks:
                result = eval_results[task]
                if task in ['cb', 'anli_r1', 'anli_r2', 'anli_r3', 'mnli']:
                    accuracies.append(result.get('f1', 0))
                else:
                    accuracies.append(result.get('accuracy', 0))

            avg_score = sum(accuracies) / len(accuracies) if accuracies else 0
            print_rank0(f"Average score: {avg_score:.4f}")
            print_rank0(f"Tasks completed: {len(valid_tasks)}/{len(all_tasks)}")
    
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
