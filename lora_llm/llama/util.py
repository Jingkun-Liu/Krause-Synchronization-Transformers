import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import DataCollatorForLanguageModeling
from transformers.models.llama.modeling_llama import LlamaAttention
from module import KrauseLlamaAttention


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_rank0(*args, **kwargs):
    GLOBAL_IS_MAIN_PROCESS = (int(os.environ.get('LOCAL_RANK', 0)) == 0)
    if GLOBAL_IS_MAIN_PROCESS:
        print(*args, **kwargs)


class ResponseOnlyDataCollator:
    
    def __init__(self, tokenizer, response_template="### Response:", mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.response_template = response_template
        self.response_template_ids_list = []
        test_contexts = [
            response_template,
            f"\n{response_template}",
            f"\n\n{response_template}",
            f" {response_template}",
        ]
        for ctx in test_contexts:
            ids = tokenizer.encode(ctx, add_special_tokens=False)
            template_ids = self._extract_template_from_context(tokenizer, ctx, response_template)
            if template_ids and template_ids not in self.response_template_ids_list:
                self.response_template_ids_list.append(template_ids)
        
        if not self.response_template_ids_list:
            full_ids = tokenizer.encode(test_contexts[2], add_special_tokens=False)
            self.response_template_ids_list.append(full_ids)
        
    
    def _extract_template_from_context(self, tokenizer, context, template):
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        template_ids = tokenizer.encode(template, add_special_tokens=False)
        
        for i in range(len(context_ids) - len(template_ids) + 1):
            if context_ids[i:i+len(template_ids)] == template_ids:
                return template_ids
        
        return context_ids
        
    def __call__(self, features):
        if not hasattr(self, '_base_collator'):
            self._base_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=self.mlm,
                pad_to_multiple_of=8,
            )
        
        batch = self._base_collator(features)
        
        input_ids = batch['input_ids']
        labels = batch['labels'].clone()
        
        for i in range(len(input_ids)):
            seq = input_ids[i]
            response_start_idx = self._find_response_start_by_text(seq)
            
            if response_start_idx is None:
                for template_ids in self.response_template_ids_list:
                    response_start_idx = self._find_response_start(seq, template_ids)
                    if response_start_idx is not None:
                        break
            
            if response_start_idx is not None:
                labels[i, :response_start_idx] = -100
        
        batch['labels'] = labels
        return batch
    
    def _find_response_start(self, sequence, template_ids):
        if len(template_ids) == 0:
            return None
        
        if hasattr(sequence, 'tolist'):
            seq_list = sequence.tolist()
        else:
            seq_list = list(sequence) if not isinstance(sequence, list) else sequence
        
        if not isinstance(template_ids, list):
            template_ids = list(template_ids)
        
        seq_len = len(seq_list)
        template_len = len(template_ids)
        
        if seq_len < template_len:
            return None
        
        for i in range(seq_len - template_len + 1):
            if seq_list[i:i+template_len] == template_ids:
                return i + template_len
        
        return None
    
    def _find_response_start_by_text(self, sequence):
        if hasattr(sequence, 'tolist'):
            seq_list = sequence.tolist()
        else:
            seq_list = list(sequence) if not isinstance(sequence, list) else sequence
        
        max_decode_len = min(512, len(seq_list))
        decoded_text = self.tokenizer.decode(seq_list[:max_decode_len], skip_special_tokens=False)
        
        template_pos = decoded_text.find(self.response_template)
        if template_pos == -1:
            return None
        
        text_upto_template_end = decoded_text[:template_pos + len(self.response_template)]
        
        best_pos = None
        for i in range(1, max_decode_len):
            decoded_i = self.tokenizer.decode(seq_list[:i], skip_special_tokens=False)
            if self.response_template in decoded_i:
                if decoded_i.endswith(self.response_template) or decoded_i.rstrip().endswith(self.response_template):
                    best_pos = i
                    break
                elif best_pos is None:
                    best_pos = i
        
        if best_pos is not None:
            return best_pos

        ids_upto_template = self.tokenizer.encode(text_upto_template_end, add_special_tokens=False)
        return len(ids_upto_template)


def run_sink_analysis(model, tokenizer, device, eval_ds, mode='baseline', max_seq_length=2048):
    
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    def find_layers_list(module):
        if hasattr(module, 'layers') and isinstance(getattr(module, 'layers'), nn.ModuleList):
            if len(module.layers) > 0 and hasattr(module.layers[0], 'self_attn'):
                 return module.layers
        
        for name, sub_module in module.named_children():
            if 'lora' in name.lower() or 'embed' in name.lower() or 'norm' in name.lower() or 'proj' in name.lower():
                continue
            
            result = find_layers_list(sub_module)
            if result is not None:
                return result
        return None

    print_rank0("\nStarting Attention Sink Analysis...")
    model.eval()
    
    layers = None
    
    if hasattr(unwrapped_model, 'base_model'):
        layers = find_layers_list(unwrapped_model.base_model.model)
    else:
        layers = find_layers_list(unwrapped_model)

    num_samples = min(500, len(eval_ds))
    print_rank0(f"   Selecting {num_samples} samples for analysis...")
    sample_ds = eval_ds.select(range(num_samples))
    
    all_layer_scores = []
    layer_paths = []
    num_layers = None
    
    for sample_idx in range(num_samples):
        try:
            text = sample_ds[sample_idx]['text']
            encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_seq_length, padding=False)
            input_ids = encoded['input_ids'].to(device)
        except Exception as e:
            print_rank0(f"Warning: Sample {sample_idx} failed to load: {e}, skipping")
            continue

        seq_len = min(input_ids.shape[1], max_seq_length)
        input_ids = input_ids[:, :seq_len]

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
    
        if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
            if num_layers is None:
                num_layers = len(outputs.attentions)
        
                all_layer_scores = [[] for _ in range(num_layers)]
                for i in range(num_layers):
                    if layers and i < len(layers):
                        layer_name = f"model.model.layers.{i}"
                        layer_paths.append(layer_name)
                    else:
                        layer_paths.append(f"layer.{i}")
            
            for i, attn_weights in enumerate(outputs.attentions):
                if attn_weights is not None:
                    if attn_weights.dim() == 4:
                        if attn_weights.shape[-2] > 1 and attn_weights.shape[-1] > 0:
                            first_token_attn = attn_weights[:, :, 1:, 0]
                            per_head_avg = first_token_attn.mean(dim=-1)
                            avg_first_token_attn = per_head_avg.mean().item()
                            all_layer_scores[i].append(avg_first_token_attn)
                    elif attn_weights.dim() == 3:
                        if attn_weights.shape[-2] > 1 and attn_weights.shape[-1] > 0:
                            first_token_attn = attn_weights[:, 1:, 0]
                            avg_first_token_attn = first_token_attn.mean().item()
                            all_layer_scores[i].append(avg_first_token_attn)
        
        elif layers is not None:
            if num_layers is None:
                num_layers = len(layers)

                all_layer_scores = [[] for _ in range(num_layers)]
                for i in range(num_layers):
                    layer_name = f"model.model.layers.{i}"
                    layer_paths.append(layer_name)
            
            for i, layer in enumerate(layers):
                attn_module = layer.self_attn
                if hasattr(attn_module, 'last_attn_weights'):
                    w = attn_module.last_attn_weights
                    if w is None:
                        continue
                    if w.dim() == 4:
                        if w.shape[-2] > 1 and w.shape[-1] > 0:
                            first_token_attn = w[:, :, 1:, 0]
                            per_head_avg = first_token_attn.mean(dim=-1)
                            avg_first_token_attn = per_head_avg.mean().item()
                            all_layer_scores[i].append(avg_first_token_attn)
                    elif w.dim() == 3:
                        if w.shape[-2] > 1 and w.shape[-1] > 0:
                            first_token_attn = w[:, 1:, 0]
                            avg_first_token_attn = first_token_attn.mean().item()
                            all_layer_scores[i].append(avg_first_token_attn)
        
        if (sample_idx + 1) % 100 == 0:
            print_rank0(f"   Processed {sample_idx + 1}/{num_samples} samples...")
    
    first_token_scores = []
    valid_samples_per_layer = []
    for layer_idx in range(len(all_layer_scores)):
        if len(all_layer_scores[layer_idx]) > 0:
            avg_score = sum(all_layer_scores[layer_idx]) / len(all_layer_scores[layer_idx])
            first_token_scores.append(avg_score)
            valid_samples_per_layer.append(len(all_layer_scores[layer_idx]))
        else:
            print_rank0(f"   Warning: Layer {layer_idx} has no valid scores")
            first_token_scores.append(0.0)
            valid_samples_per_layer.append(0)
    
    print_rank0(f"\nCompleted analysis: Processed {num_samples} samples")
    if valid_samples_per_layer:
        print_rank0(f"\nValid samples per layer: {min(valid_samples_per_layer)} - {max(valid_samples_per_layer)}")

    if first_token_scores and len(first_token_scores) > 0:
        if len(first_token_scores) != len(layers) if layers else 0:
            print_rank0(f"Warning: Only got {len(first_token_scores)} layers of attention scores, expected {len(layers) if layers else 0} layers")
        avg_sink_score = sum(first_token_scores) / len(first_token_scores)
        
        for layer_idx, score in enumerate(first_token_scores):
            layer_path = layer_paths[layer_idx] if layer_idx < len(layer_paths) else f"model.model.layers.{layer_idx}"
            print_rank0(f"Layer {layer_idx} ({layer_path}): sink={score:.6f}")
        
        print_rank0(f"====================================================")
        print_rank0(f"\n Sink Metric Summary")
        print_rank0(f"global average sink: {avg_sink_score:.5f}")
        print_rank0(f"====================================================")
        
        threshold = 0.05
        abnormal_layers = []
        for layer_idx, score in enumerate(first_token_scores):
            if score > threshold:
                layer_path = layer_paths[layer_idx] if layer_idx < len(layer_paths) else f"model.model.layers.{layer_idx}"
                abnormal_layers.append((layer_idx, layer_path, score))
        
        if abnormal_layers:
            print_rank0(f"\nDetected layers with abnormal Sink behavior:")
            for layer_idx, layer_path, score in abnormal_layers:
                print_rank0(f"   - Layer {layer_idx:02d} ({layer_path}): {score:.6f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        layer_indices = list(range(len(first_token_scores)))
        
        ax.plot(layer_indices, first_token_scores, 'b-', linewidth=2, label='first token attention score')
        
        mean_score = avg_sink_score
        ax.axhline(y=mean_score, color='g', linestyle='--', linewidth=2, label=f'mean: {mean_score:.3f}')
        
        ax.set_xlabel('layer', fontsize=12)
        ax.set_ylabel('first token attention score', fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, len(first_token_scores) - 1)
        
        num_layers = len(first_token_scores)
        if num_layers <= 32:
            step = max(1, num_layers // 6)
            ax.set_xticks(range(0, num_layers, step))
        else:
            ax.set_xticks(range(0, num_layers, max(1, num_layers // 6)))
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        
        title = f'first token attention scores of {mode}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        output_path = f'sink_metric_{mode}_llama3_8b.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_rank0(f"Image saved: {output_path}")

def replace_attention_modules(model, mode: str, krause_params: dict):
    print_rank0(f"Replacing model Attention → mode: {mode.upper()} (dual branch, token-level softmax gate mixed)")
    
    if mode != 'krause':
        raise ValueError(f"Invalid mode: {mode}. Only 'krause' mode is supported.")
    
    target_cls = KrauseLlamaAttention

    replaced_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LlamaAttention):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            
            layer_idx = module.layer_idx
            
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype

            new_module = target_cls(module.config, krause_params, layer_idx=layer_idx).to(device).to(dtype)

            with torch.no_grad():
                new_module.q_proj.weight.copy_(module.q_proj.weight)
                new_module.k_proj.weight.copy_(module.k_proj.weight)
                new_module.v_proj.weight.copy_(module.v_proj.weight)
                new_module.o_proj.weight.copy_(module.o_proj.weight)
                
                if module.q_proj.bias is not None: new_module.q_proj.bias.copy_(module.q_proj.bias)
                if module.k_proj.bias is not None: new_module.k_proj.bias.copy_(module.k_proj.bias)
                if module.v_proj.bias is not None: new_module.v_proj.bias.copy_(module.v_proj.bias)

                new_module.q_proj_krause.weight.copy_(module.q_proj.weight)
                new_module.k_proj_krause.weight.copy_(module.k_proj.weight)
                new_module.v_proj_krause.weight.copy_(module.v_proj.weight)
                new_module.o_proj_krause.weight.copy_(module.o_proj.weight)
                
                if module.q_proj.bias is not None: new_module.q_proj_krause.bias.copy_(module.q_proj.bias)
                if module.k_proj.bias is not None: new_module.k_proj_krause.bias.copy_(module.k_proj.bias)
                if module.v_proj.bias is not None: new_module.v_proj_krause.bias.copy_(module.v_proj.bias)

            setattr(parent, child_name, new_module)
            replaced_count += 1

    print_rank0(f"\nTotal replaced Attention layers (dual branch): {replaced_count}")
