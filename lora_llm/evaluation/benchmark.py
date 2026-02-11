import os
import json
import traceback
import time
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score

from util import print_rank0, sync_distributed_results, GLOBAL_LOCAL_RANK, GLOBAL_IS_MAIN_PROCESS

def evaluate_choice_task_distributed(model, tokenizer, dataset, device, task_name="ARC", batch_size=4):
    model.eval()
    local_correct = 0
    local_total = 0

    choice_ids = []
    for c in ["A", "B", "C", "D"]:
        encoded = tokenizer.encode(c, add_special_tokens=False)
        if len(encoded) == 1:
            choice_ids.append(encoded[0])
        elif len(encoded) > 1:
            choice_ids.append(encoded[-1])
        else:
            encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
            if len(encoded_space) > 0:
                choice_ids.append(encoded_space[-1])
            else:
                encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                if len(encoded_full) > 0:
                    choice_ids.append(encoded_full[0])
                else:
                    raise ValueError(f"Cannot encode choice {c}")
    
    def collate_fn(examples):
        if not examples:
            return {}
        keys = examples[0].keys()
        batch = {}
        for key in keys:
            batch[key] = [ex[key] for ex in examples]
        return batch
    
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    if dist.is_initialized():
        actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
    else:
        actual_device = device if isinstance(device, torch.device) else torch.device(device)

    for batch in tqdm(dataloader, desc=f"Eval {task_name}", disable=not GLOBAL_IS_MAIN_PROCESS):
        prompts = []
        labels = []
        
        if not isinstance(batch, dict):
            continue
            
        if 'question' not in batch or 'choices' not in batch or 'answer' not in batch:
            continue
        
        questions = batch['question']
        choices_batch = batch['choices']
        answers = batch['answer']
        
        batch_size = len(questions)
        if batch_size == 0:
            continue
        
        for i in range(batch_size):
            try:
                if i >= len(questions) or i >= len(choices_batch) or i >= len(answers):
                    continue

                question = questions[i]
                choices_list = choices_batch[i]

                if not isinstance(choices_list, list):
                    continue
                while len(choices_list) < 4:
                    choices_list.append('')
                choices_list = choices_list[:4]

                instruction = f"Question: {question}\nChoices:\nA. {choices_list[0]}\nB. {choices_list[1]}\nC. {choices_list[2]}\nD. {choices_list[3]}"
                p = f"{instruction}\n\n### Response:\n"
                
                prompts.append(p)
                
                answer = answers[i]
                try:
                    if answer is None or (isinstance(answer, str) and answer.strip() == ''):
                        labels.append(-1)
                        continue
                    
                    answer_idx = -1
                    if isinstance(answer, str):
                        answer_upper = answer.strip().upper()
                        if answer_upper in ['A', 'B', 'C', 'D']:
                            answer_idx = ord(answer_upper) - ord('A')
                        elif answer_upper.isdigit():
                            answer_idx = int(answer_upper)
                            if answer_idx < 0 or answer_idx > 3:
                                answer_idx = -1
                        else:
                            try:
                                answer_idx = int(answer_upper)
                                if answer_idx < 0 or answer_idx > 3:
                                    answer_idx = -1
                            except:
                                answer_idx = -1
                    else:
                        answer_idx = int(answer)
                        if answer_idx < 0 or answer_idx > 3:
                            answer_idx = -1
                    
                    if 0 <= answer_idx <= 3:
                        labels.append(answer_idx)
                    else:
                        labels.append(-1)
                except (ValueError, TypeError, AttributeError) as e:
                    labels.append(-1)
            except (IndexError, KeyError) as e:
                continue
        
        min_len = min(len(prompts), len(labels))
        if min_len == 0:
            continue

        prompts = prompts[:min_len]
        labels = labels[:min_len]

        valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 3]
        if not valid_indices:
            continue
        
        prompts = [prompts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                last_token_logits = logits[:, -1, :]
                choice_logits = last_token_logits[:, choice_ids]
                preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()

            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    local_correct += 1
                local_total += 1
        except Exception as e:
            print_rank0(f"Error processing batch: {e}")
            continue

    total_correct, total_all = sync_distributed_results(local_correct, local_total)
    return {"accuracy": total_correct / total_all if total_all > 0 else 0, "correct": total_correct, "total": total_all}


def evaluate_choice_task_2option_distributed(model, tokenizer, dataset, device, task_name="COPA", batch_size=4):
    model.eval()
    local_correct = 0
    local_total = 0

    choice_ids = []
    for c in ["A", "B"]:
        encoded = tokenizer.encode(c, add_special_tokens=False)
        if len(encoded) == 1:
            choice_ids.append(encoded[0])
        elif len(encoded) > 1:
            choice_ids.append(encoded[-1])
        else:
            encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
            if len(encoded_space) > 0:
                choice_ids.append(encoded_space[-1])
            else:
                encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                if len(encoded_full) > 0:
                    choice_ids.append(encoded_full[0])
                else:
                    raise ValueError(f"Cannot encode choice {c}")
    
    def collate_fn(examples):
        if not examples:
            return {}
        keys = examples[0].keys()
        batch = {}
        for key in keys:
            batch[key] = [ex[key] for ex in examples]
        return batch
    
    sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    if dist.is_initialized():
        actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
    else:
        actual_device = device if isinstance(device, torch.device) else torch.device(device)

    for batch in tqdm(dataloader, desc=f"Eval {task_name}", disable=not GLOBAL_IS_MAIN_PROCESS):
        prompts = []
        labels = []
        
        if not isinstance(batch, dict):
            continue
            
        if 'question' not in batch or 'choices' not in batch or 'answer' not in batch:
            continue
        
        questions = batch['question']
        choices_batch = batch['choices']
        answers = batch['answer']
        
        batch_size = len(questions)
        if batch_size == 0:
            continue
        
        for i in range(batch_size):
            try:
                if i >= len(questions) or i >= len(choices_batch) or i >= len(answers):
                    continue
                
                question = questions[i]
                choices_list = choices_batch[i]
                
                if not isinstance(choices_list, list):
                    continue
                if len(choices_list) < 2:
                    continue
                
                instruction = f"Question: {question}\nA. {choices_list[0]}\nB. {choices_list[1]}"
                p = f"{instruction}\n\n### Response:\n"
                prompts.append(p)
                
                answer = answers[i]
                try:
                    if isinstance(answer, str):
                        answer_upper = answer.strip().upper()
                        if answer_upper == 'A':
                            answer_idx = 0
                        elif answer_upper == 'B':
                            answer_idx = 1
                        else:
                            answer_num = int(answer_upper)
                            if answer_num == 0:
                                answer_idx = 0
                            elif answer_num == 1:
                                answer_idx = 1
                            else:
                                labels.append(-1)
                                continue
                    else:
                        answer_num = int(answer)
                        if answer_num == 0:
                            answer_idx = 0
                        elif answer_num == 1:
                            answer_idx = 1
                        else:
                            labels.append(-1)
                            continue
                    
                    if 0 <= answer_idx <= 1:
                        labels.append(answer_idx)
                    else:
                        labels.append(-1)
                except (ValueError, TypeError):
                    labels.append(-1)
            except (IndexError, KeyError):
                continue
        
        min_len = min(len(prompts), len(labels))
        if min_len == 0:
            continue
        
        prompts = prompts[:min_len]
        labels = labels[:min_len]
        
        valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 1]
        if not valid_indices:
            continue
        
        prompts = [prompts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                last_token_logits = logits[:, -1, :]
                
                choice_logits = last_token_logits[:, choice_ids]
                preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
            
            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    local_correct += 1
                local_total += 1
        except Exception as e:
            print_rank0(f"Error processing batch: {e}")
            continue

    total_correct, total_all = sync_distributed_results(local_correct, local_total)
    return {"accuracy": total_correct / total_all if total_all > 0 else 0, "correct": total_correct, "total": total_all}


def evaluate_boolq(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/boolq/data"
        val_path = os.path.join(local_root, "validation-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")

        model.eval()
        local_correct = 0
        local_total = 0
        
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[-1]
        
        if GLOBAL_IS_MAIN_PROCESS:
            print_rank0(f"Target IDs -> Yes: {yes_token_id} ('{tokenizer.decode([yes_token_id])}'), "
                        f"No: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()})
        
        actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}') if dist.is_initialized() else device

        start_time = time.time()
        batch_count = 0

        for batch in tqdm(dataloader, desc="Eval BoolQ", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            for i in range(len(batch['passage'])):
                instruction = f"Passage: {batch['passage'][i]}\nQuestion: {batch['question'][i]}?"
                prompt = f"{instruction}\n\n### Response:\n"
                prompts.append(prompt)
                
                label = batch['label'][i]
                labels.append(1 if (label is True or label == 1) else 0)
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=1024).to(actual_device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits # [batch, seq_len, vocab_size]
                
                last_token_logits = logits[:, -1, :] 
                
                pos_logits = last_token_logits[:, yes_token_id]
                neg_logits = last_token_logits[:, no_token_id]
                
                preds = (pos_logits > neg_logits).long().cpu().numpy()
                
                for p, l in zip(preds, labels):
                    if p == l: local_correct += 1
                    local_total += 1
            
            batch_count += 1
                        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        end_time = time.time()
        total_time = end_time - start_time
        if total_time > 0 and batch_count > 0:
            avg_it_per_sec = batch_count / total_time
            batch_size = 4
            avg_sample_per_sec = avg_it_per_sec * batch_size
            if GLOBAL_IS_MAIN_PROCESS:
                print_rank0(f"BoolQ avg speed: {avg_sample_per_sec:.2f} sample/s")
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, "correct": total_correct, "total": total_all}
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def evaluate_cb(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/cb"
        val_path = os.path.join(local_root, "validation-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        model.eval()
        local_correct = 0
        local_total = 0
        local_predictions = []
        local_labels = []
        
        choice_ids = []
        for c in ["A", "B", "C"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval CB", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict) or 'premise' not in batch or 'hypothesis' not in batch or 'label' not in batch:
                continue
            
            premises = batch['premise']
            hypotheses = batch['hypothesis']
            label_items = batch['label']
            
            batch_size = len(premises)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    premise = premises[i]
                    hypothesis = hypotheses[i]
                    label_item = label_items[i]
                    
                    instruction = f"Premise: {premise}\nHypothesis: {hypothesis}\nA. entailment\nB. contradiction\nC. neutral"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    try:
                        if isinstance(label_item, str):
                            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                            label_idx = label_map.get(label_item.lower(), -1)
                        else:
                            label_idx = int(label_item) if 0 <= int(label_item) <= 2 else -1
                        
                        if 0 <= label_idx <= 2:
                            labels.append(label_idx)
                        else:
                            labels.append(-1)
                    except (ValueError, TypeError):
                        labels.append(-1)
                except (IndexError, KeyError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 2]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    last_token_logits = logits[:, -1, :]
                    
                    choice_logits = last_token_logits[:, choice_ids]
                    preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                    
                    local_predictions.extend(preds.tolist())
                    local_labels.extend(labels)
                    
                    for i in range(len(preds)):
                        if preds[i] == labels[i]:
                            local_correct += 1
                        local_total += 1
            except Exception as e:
                print_rank0(f"Error processing batch: {e}")
                continue
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        if dist.is_initialized():
            all_predictions = [None] * dist.get_world_size()
            all_labels = [None] * dist.get_world_size()
            dist.all_gather_object(all_predictions, local_predictions)
            dist.all_gather_object(all_labels, local_labels)
            
            all_predictions_flat = []
            all_labels_flat = []
            for pred_list, label_list in zip(all_predictions, all_labels):
                all_predictions_flat.extend(pred_list)
                all_labels_flat.extend(label_list)
        else:
            all_predictions_flat = local_predictions
            all_labels_flat = local_labels
        
        f1 = 0.0
        if len(all_predictions_flat) > 0 and len(all_labels_flat) > 0:
            f1 = f1_score(all_labels_flat, all_predictions_flat, average='macro', zero_division=0)
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, 
                "f1": f1,
                "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_anli_r1(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/anli"
        val_path = os.path.join(local_root, "dev_r1-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        model.eval()
        local_correct = 0
        local_total = 0
        local_predictions = []
        local_labels = []

        choice_ids = []
        for c in ["A", "B", "C"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval ANLI-R1", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict) or 'premise' not in batch or 'hypothesis' not in batch or 'label' not in batch:
                continue
            
            premises = batch['premise']
            hypotheses = batch['hypothesis']
            label_items = batch['label']
            
            batch_size = len(premises)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    premise = premises[i]
                    hypothesis = hypotheses[i]
                    label_item = label_items[i]
                    
                    instruction = f"Premise: {premise}\nHypothesis: {hypothesis}\nA. entailment\nB. contradiction\nC. neutral"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    try:
                        if isinstance(label_item, str):
                            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                            label_idx = label_map.get(label_item.lower(), -1)
                        else:
                            label_idx = int(label_item) if 0 <= int(label_item) <= 2 else -1
                        
                        if 0 <= label_idx <= 2:
                            labels.append(label_idx)
                        else:
                            labels.append(-1)
                    except (ValueError, TypeError):
                        labels.append(-1)
                except (IndexError, KeyError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 2]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    last_token_logits = logits[:, -1, :]
                    
                    choice_logits = last_token_logits[:, choice_ids]
                    preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                    
                    local_predictions.extend(preds.tolist())
                    local_labels.extend(labels)
                    
                    for i in range(len(preds)):
                        if preds[i] == labels[i]:
                            local_correct += 1
                        local_total += 1
            except Exception as e:
                print_rank0(f"Error processing batch: {e}")
                continue
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        if dist.is_initialized():
            all_predictions = [None] * dist.get_world_size()
            all_labels = [None] * dist.get_world_size()
            dist.all_gather_object(all_predictions, local_predictions)
            dist.all_gather_object(all_labels, local_labels)
            
            all_predictions_flat = []
            all_labels_flat = []
            for pred_list, label_list in zip(all_predictions, all_labels):
                all_predictions_flat.extend(pred_list)
                all_labels_flat.extend(label_list)
        else:
            all_predictions_flat = local_predictions
            all_labels_flat = local_labels
        
        f1 = 0.0
        if len(all_predictions_flat) > 0 and len(all_labels_flat) > 0:
            f1 = f1_score(all_labels_flat, all_predictions_flat, average='macro', zero_division=0)
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, 
                "f1": f1,
                "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_anli_r2(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/anli"
        val_path = os.path.join(local_root, "dev_r2-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        model.eval()
        local_correct = 0
        local_total = 0
        local_predictions = []
        local_labels = []

        choice_ids = []
        for c in ["A", "B", "C"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval ANLI-R2", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict) or 'premise' not in batch or 'hypothesis' not in batch or 'label' not in batch:
                continue
            
            premises = batch['premise']
            hypotheses = batch['hypothesis']
            label_items = batch['label']
            
            batch_size = len(premises)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    premise = premises[i]
                    hypothesis = hypotheses[i]
                    label_item = label_items[i]
                    
                    instruction = f"Premise: {premise}\nHypothesis: {hypothesis}\nA. entailment\nB. contradiction\nC. neutral"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    try:
                        if isinstance(label_item, str):
                            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                            label_idx = label_map.get(label_item.lower(), -1)
                        else:
                            label_idx = int(label_item) if 0 <= int(label_item) <= 2 else -1
                        
                        if 0 <= label_idx <= 2:
                            labels.append(label_idx)
                        else:
                            labels.append(-1)
                    except (ValueError, TypeError):
                        labels.append(-1)
                except (IndexError, KeyError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 2]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    last_token_logits = logits[:, -1, :]
                    
                    choice_logits = last_token_logits[:, choice_ids]
                    preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                    
                    local_predictions.extend(preds.tolist())
                    local_labels.extend(labels)
                    
                    for i in range(len(preds)):
                        if preds[i] == labels[i]:
                            local_correct += 1
                        local_total += 1
            except Exception as e:
                print_rank0(f"Error processing batch: {e}")
                continue
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        if dist.is_initialized():
            all_predictions = [None] * dist.get_world_size()
            all_labels = [None] * dist.get_world_size()
            dist.all_gather_object(all_predictions, local_predictions)
            dist.all_gather_object(all_labels, local_labels)
            
            all_predictions_flat = []
            all_labels_flat = []
            for pred_list, label_list in zip(all_predictions, all_labels):
                all_predictions_flat.extend(pred_list)
                all_labels_flat.extend(label_list)
        else:
            all_predictions_flat = local_predictions
            all_labels_flat = local_labels
        
        f1 = 0.0
        if len(all_predictions_flat) > 0 and len(all_labels_flat) > 0:
            f1 = f1_score(all_labels_flat, all_predictions_flat, average='macro', zero_division=0)
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, 
                "f1": f1,
                "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_anli_r3(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/anli"
        val_path = os.path.join(local_root, "dev_r3-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        model.eval()
        local_correct = 0
        local_total = 0
        local_predictions = []
        local_labels = []

        choice_ids = []
        for c in ["A", "B", "C"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval ANLI-R3", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict) or 'premise' not in batch or 'hypothesis' not in batch or 'label' not in batch:
                continue
            
            premises = batch['premise']
            hypotheses = batch['hypothesis']
            label_items = batch['label']
            
            batch_size = len(premises)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    premise = premises[i]
                    hypothesis = hypotheses[i]
                    label_item = label_items[i]
                    
                    instruction = f"Premise: {premise}\nHypothesis: {hypothesis}\nA. entailment\nB. contradiction\nC. neutral"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    try:
                        if isinstance(label_item, str):
                            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                            label_idx = label_map.get(label_item.lower(), -1)
                        else:
                            label_idx = int(label_item) if 0 <= int(label_item) <= 2 else -1
                        
                        if 0 <= label_idx <= 2:
                            labels.append(label_idx)
                        else:
                            labels.append(-1)
                    except (ValueError, TypeError):
                        labels.append(-1)
                except (IndexError, KeyError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 2]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    last_token_logits = logits[:, -1, :]
                    
                    choice_logits = last_token_logits[:, choice_ids]
                    preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                    
                    local_predictions.extend(preds.tolist())
                    local_labels.extend(labels)
                    
                    for i in range(len(preds)):
                        if preds[i] == labels[i]:
                            local_correct += 1
                        local_total += 1
            except Exception as e:
                print_rank0(f"Error processing batch: {e}")
                continue
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        if dist.is_initialized():
            all_predictions = [None] * dist.get_world_size()
            all_labels = [None] * dist.get_world_size()
            dist.all_gather_object(all_predictions, local_predictions)
            dist.all_gather_object(all_labels, local_labels)
            
            all_predictions_flat = []
            all_labels_flat = []
            for pred_list, label_list in zip(all_predictions, all_labels):
                all_predictions_flat.extend(pred_list)
                all_labels_flat.extend(label_list)
        else:
            all_predictions_flat = local_predictions
            all_labels_flat = local_labels
        
        f1 = 0.0
        if len(all_predictions_flat) > 0 and len(all_labels_flat) > 0:
            f1 = f1_score(all_labels_flat, all_predictions_flat, average='macro', zero_division=0)
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, 
                "f1": f1,
                "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_piqa(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/piqa"
        val_path = os.path.join(local_root, "piqa_validation.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        def convert_piqa_format(example):
            goal_key = 'goal' if 'goal' in example else 'question'
            sol1_key = 'sol1'
            sol2_key = 'sol2'
            label_key = 'label' if 'label' in example else 'answer'
            
            question = example.get(goal_key, '')
            choices = [example.get(sol1_key, ''), example.get(sol2_key, '')]
            
            label_item = example.get(label_key, -1)
            try:
                if isinstance(label_item, (int, float)):
                    answer = int(label_item)
                else:
                    label_str = str(label_item).strip()
                    if label_str in ['0', 'A', 'a']:
                        answer = 0
                    elif label_str in ['1', 'B', 'b']:
                        answer = 1
                    else:
                        answer = -1
            except (ValueError, TypeError):
                answer = -1
            
            return {"question": question, "choices": choices, "answer": answer}
        
        dataset = dataset.map(convert_piqa_format).filter(lambda x: x["answer"] >= 0)
        return evaluate_choice_task_2option_distributed(model, tokenizer, dataset, device, "PIQA", batch_size=4)
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_mnli(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/mnli"
        val_path = os.path.join(local_root, "validation_matched.jsonl")
        
        dataset_list = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset_list.append(json.loads(line.strip()))

        dataset = Dataset.from_list(dataset_list)
        
        model.eval()
        local_correct = 0
        local_total = 0
        local_predictions = []
        local_labels = []

        choice_ids = []
        for c in ["A", "B", "C"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval MNLI", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict) or 'text1' not in batch or 'text2' not in batch or 'label' not in batch:
                continue
            
            text1s = batch['text1']
            text2s = batch['text2']
            label_items = batch['label']
            
            batch_size = len(text1s)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    text1 = text1s[i]
                    text2 = text2s[i]
                    label_item = label_items[i]
                    
                    instruction = f"Premise: {text1}\nHypothesis: {text2}\nA. entailment\nB. contradiction\nC. neutral"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    try:
                        if isinstance(label_item, str):
                            label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
                            label_idx = label_map.get(label_item.lower(), -1)
                        else:
                            label_idx = int(label_item) if 0 <= int(label_item) <= 2 else -1
                            if label_idx == 0:
                                label_idx = 0
                            elif label_idx == 1:
                                label_idx = 2
                            elif label_idx == 2:
                                label_idx = 1
                        
                        if 0 <= label_idx <= 2:
                            labels.append(label_idx)
                        else:
                            labels.append(-1)
                    except (ValueError, TypeError):
                        labels.append(-1)
                except (IndexError, KeyError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 2]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    last_token_logits = logits[:, -1, :]
                    
                    choice_logits = last_token_logits[:, choice_ids]
                    preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                    
                    local_predictions.extend(preds.tolist())
                    local_labels.extend(labels)
                    
                    for i in range(len(preds)):
                        if preds[i] == labels[i]:
                            local_correct += 1
                        local_total += 1
            except Exception as e:
                print_rank0(f"Error processing batch: {e}")
                continue
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        
        if dist.is_initialized():
            all_predictions = [None] * dist.get_world_size()
            all_labels = [None] * dist.get_world_size()
            dist.all_gather_object(all_predictions, local_predictions)
            dist.all_gather_object(all_labels, local_labels)
            
            all_predictions_flat = []
            all_labels_flat = []
            for pred_list, label_list in zip(all_predictions, all_labels):
                all_predictions_flat.extend(pred_list)
                all_labels_flat.extend(label_list)
        else:
            all_predictions_flat = local_predictions
            all_labels_flat = local_labels
        
        f1 = 0.0
        if len(all_predictions_flat) > 0 and len(all_labels_flat) > 0:
            f1 = f1_score(all_labels_flat, all_predictions_flat, average='macro', zero_division=0)
        
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, 
                "f1": f1,
                "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_mmlupro(model, tokenizer, device):
    try:
        local_root = "./lora_datasets/datasets/mmlupro"
        val_path = os.path.join(local_root, "validation-00000-of-00001.parquet")
        dataset = load_dataset("parquet", data_files=val_path, split="train")
        
        model.eval()
        local_correct = 0
        local_total = 0
        
        choice_ids = []
        for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            encoded = tokenizer.encode(c, add_special_tokens=False)
            if len(encoded) == 1:
                choice_ids.append(encoded[0])
            elif len(encoded) > 1:
                choice_ids.append(encoded[-1])
            else:
                encoded_space = tokenizer.encode(f" {c}", add_special_tokens=False)
                if len(encoded_space) > 0:
                    choice_ids.append(encoded_space[-1])
                else:
                    encoded_full = tokenizer.encode(f"{c}.", add_special_tokens=False)
                    if len(encoded_full) > 0:
                        choice_ids.append(encoded_full[0])
                    else:
                        raise ValueError(f"Cannot encode choice {c}")
        
        def collate_fn(examples):
            if not examples:
                return {}
            keys = examples[0].keys()
            batch = {}
            for key in keys:
                batch[key] = [ex[key] for ex in examples]
            return batch
        
        sampler = DistributedSampler(dataset, shuffle=False) if dist.is_initialized() else None
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        if dist.is_initialized():
            actual_device = torch.device(f'cuda:{GLOBAL_LOCAL_RANK}')
        else:
            actual_device = device if isinstance(device, torch.device) else torch.device(device)
        
        for batch in tqdm(dataloader, desc="Eval MMLU-Pro", disable=not GLOBAL_IS_MAIN_PROCESS):
            prompts = []
            labels = []
            
            if not isinstance(batch, dict):
                continue
            
            question_key = 'question' if 'question' in batch else None
            options_key = 'options' if 'options' in batch else None
            answer_key = 'answer' if 'answer' in batch else None
            answer_index_key = 'answer_index' if 'answer_index' in batch else None
            
            if question_key is None or options_key is None:
                if GLOBAL_IS_MAIN_PROCESS:
                    print_rank0(f"MMLU-Pro batch field mismatch. Available: {list(batch.keys())}")
                continue
            
            questions = batch[question_key]
            options_batch = batch[options_key]
            answers = batch[answer_key] if answer_key else None
            answer_indices = batch[answer_index_key] if answer_index_key else None
            
            batch_size = len(questions)
            if batch_size == 0:
                continue
            
            for i in range(batch_size):
                try:
                    if i >= len(questions) or i >= len(options_batch):
                        continue
                    
                    question = questions[i]
                    options_item = options_batch[i]
                    
                    options_list = []
                    if isinstance(options_item, list):
                        for option in options_item:
                            options_list.append(str(option))
                    else:
                        continue
                    
                    if len(options_list) < 10:
                        while len(options_list) < 10:
                            options_list.append('')
                    options_list = options_list[:10]
                    
                    instruction = f"Question: {question}\nChoices:\nA. {options_list[0]}\nB. {options_list[1]}\nC. {options_list[2]}\nD. {options_list[3]}\nE. {options_list[4]}\nF. {options_list[5]}\nG. {options_list[6]}\nH. {options_list[7]}\nI. {options_list[8]}\nJ. {options_list[9]}"
                    p = f"{instruction}\n\n### Response:\n"
                    prompts.append(p)
                    
                    label_idx = -1
                    if answer_index_key and answer_indices is not None:
                        try:
                            idx = int(answer_indices[i])
                            if 0 <= idx <= 9:
                                label_idx = idx
                        except (ValueError, TypeError, IndexError):
                            pass
                    
                    if label_idx == -1 and answer_key and answers is not None:
                        try:
                            answer_str = str(answers[i]).strip().upper()
                            if answer_str in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                                label_idx = ord(answer_str) - ord('A')
                        except (ValueError, TypeError, IndexError):
                            pass
                    
                    if 0 <= label_idx <= 9:
                        labels.append(label_idx)
                    else:
                        labels.append(-1)
                except (IndexError, KeyError, AttributeError):
                    continue
            
            min_len = min(len(prompts), len(labels))
            if min_len == 0:
                continue
            
            prompts = prompts[:min_len]
            labels = labels[:min_len]
            
            valid_indices = [i for i, label in enumerate(labels) if 0 <= label <= 9]
            if not valid_indices:
                continue
            
            prompts = [prompts[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(actual_device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                last_token_logits = logits[:, -1, :]
                
                choice_logits = last_token_logits[:, choice_ids]
                preds = torch.argmax(choice_logits, dim=-1).cpu().numpy()
                
                for i in range(len(preds)):
                    if preds[i] == labels[i]:
                        local_correct += 1
                    local_total += 1
        
        total_correct, total_all = sync_distributed_results(local_correct, local_total)
        return {"accuracy": total_correct / total_all if total_all > 0 else 0, "correct": total_correct, "total": total_all}
    
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
