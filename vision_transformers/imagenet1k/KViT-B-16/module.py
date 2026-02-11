import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.transforms.v2 as transforms_v2
import torch.distributed as dist
import sys


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
        sys.stdout.flush()


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, start_lr=0.0, target_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr if target_lr is not None else optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
    def step(self):
        if self.current_step < self.warmup_steps:
            lr_scale = (self.current_step + 1) / self.warmup_steps
            lr = self.start_lr + (self.target_lr - self.start_lr) * lr_scale
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_step += 1
    
    def is_warming_up(self):
        return self.current_step < self.warmup_steps


def drop_path(x, drop_prob: float = 0.0, training: bool = False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class KrauseViTAttention(nn.Module):
    def __init__(self, d_model, n_heads, sigma=1.0, top_k=32, dropout=0.0, grid_size=14):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.top_k = top_k
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.grid_size = grid_size
        self.n_patches = grid_size * grid_size
        self.n_tokens = self.n_patches + 1
        
        locality_mask = self.create_locality_mask(self.n_tokens, self.grid_size)
        self.register_buffer('local_mask_buffer', locality_mask)
        
        valid_indices = torch.nonzero(locality_mask, as_tuple=True)
        self.register_buffer('valid_i_indices', valid_indices[0])
        self.register_buffer('valid_j_indices', valid_indices[1])
        
    def create_locality_mask(self, n_tokens, grid_size):
        mask = torch.zeros(n_tokens, n_tokens, dtype=torch.bool)
        
        mask[0, :] = True
        mask[:, 0] = True
        
        for i in range(1, n_tokens):
            for j in range(1, n_tokens):
                if i == j:
                    mask[i, j] = True
                    continue
                
                r1, c1 = (i-1) // grid_size, (i-1) % grid_size
                r2, c2 = (j-1) // grid_size, (j-1) % grid_size
                
                if max(abs(r1-r2), abs(c1-c2)) <= 2:
                    mask[i, j] = True
        
        return mask
    
    def rbf_attention_weights(self, query, key):
        B, H, N, D = query.shape
        sigma_sq = torch.exp(2 * self.log_sigma)
        
        i_idx = self.valid_i_indices
        j_idx = self.valid_j_indices
        
        q_valid = query[:, :, i_idx, :]
        k_valid = key[:, :, j_idx, :]
        
        diff = q_valid - k_valid
        dist_sq_valid = torch.sum(diff**2, dim=-1)
        
        scores_valid = -0.5 / sigma_sq * dist_sq_valid
        
        scores = torch.full((B, H, N, N), float('-inf'), device=query.device)
        
        batch_idx = torch.arange(B, device=query.device).view(B, 1, 1).expand(B, H, len(i_idx))
        head_idx = torch.arange(H, device=query.device).view(1, H, 1).expand(B, H, len(i_idx))
        
        i_idx_expanded = i_idx.view(1, 1, -1).expand(B, H, -1)
        j_idx_expanded = j_idx.view(1, 1, -1).expand(B, H, -1)
        
        scores[batch_idx, head_idx, i_idx_expanded, j_idx_expanded] = scores_valid
        
        current_top_k = min(self.top_k, N) 
        
        top_k_values, top_k_indices = torch.topk(scores, k=current_top_k, dim=-1, largest=True)
        
        full_topk_mask = torch.zeros((B, H, N, N), dtype=torch.bool, device=query.device)
        full_topk_mask.scatter_(-1, top_k_indices, True)
        
        valid_scores_mask = scores != float('-inf')
        final_mask = valid_scores_mask & full_topk_mask
        
        scores = scores.masked_fill(~final_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        
        attention_mask = (weights > 1e-6).float()

        return weights, attention_mask
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_weights, attention_mask = self.rbf_attention_weights(Q, K)
        attn_weights = self.dropout(attn_weights)
        
        consensus = torch.matmul(attn_weights.to(V.dtype), V)
        
        consensus = consensus.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(consensus)
        
        return out, attn_weights, attention_mask


def count_krause_attention_flops(m, input, output=None):
    
    N = m.n_tokens
    D = m.d_model
    H = m.n_heads
    D_h = m.d_k
    K = m.top_k
    L = len(m.valid_i_indices)
    
    total_flops = 0
    
    qkv_flops = 3 * N * (2 * D - 1) * D
    total_flops += qkv_flops
    
    rbf_flops = H * L * (3 * D_h + 1)
    total_flops += rbf_flops
    
    softmax_sparse_elements = N * min(K, N)
    softmax_flops = H * softmax_sparse_elements * 5
    total_flops += softmax_flops
    
    sparse_matmul_flops = H * softmax_sparse_elements * (2 * D_h - 1)
    total_flops += sparse_matmul_flops
    
    proj_flops = N * (2 * D - 1) * D
    total_flops += proj_flops
        
    total_params = sum(p.numel() for p in m.parameters())

    return int(total_flops), int(total_params)


class KrauseViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, sigma=1.0, top_k=5,
                 dropout=0.1, grid_size=14, drop_path=0.0):
        super().__init__()
        
        self.attention = KrauseViTAttention(
            d_model, n_heads, sigma, top_k, dropout, grid_size=grid_size
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        
        try:
            attn_out, attn_weights, attention_mask = self.attention(x)
        except ValueError:
            attention_outputs = self.attention(x)
            
            if len(attention_outputs) >= 2:
                attn_out, attn_weights = attention_outputs[:2]
            elif len(attention_outputs) == 1:
                 attn_out = attention_outputs[0]
                 attn_weights = None
            else:
                 raise RuntimeError("KrauseViTAttention returned unexpected number of outputs during FLOPs calculation.")
                 
            attention_mask = None
            
        x = x + self.drop_path(attn_out)
        x = self.norm1(x)
        
        ffn_out = self.ffn(x)
        x = x + self.drop_path(ffn_out)
        x = self.norm2(x)
        
        return x, attn_weights, attention_mask


class KrauseVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 d_model=768, n_heads=12, n_layers=12, d_ff=3072,
                 sigma=1.0, top_k=32, dropout=0.0, drop_path=0.0): # ImageNet ViT-B/16 defaults
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.n_patches = self.patch_embed.n_patches
        
        self.grid_size = img_size // patch_size
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        
        top_k_min_val = top_k
        top_k_max_val = 2 * top_k
        
        if n_layers > 1:
            top_k_values = torch.linspace(top_k_min_val, top_k_max_val, n_layers).round().int().tolist()
        else:
            top_k_values = [top_k_min_val]
        
        print_rank0(f"Krause-ViT: Initializing with {n_layers} layers and varying top_k from {top_k_min_val} to {top_k_max_val}: {top_k_values}")

        if n_layers > 1:
            dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]
        else:
            dpr = [drop_path]
        
        self.blocks = nn.ModuleList([
            KrauseViTBlock(d_model, n_heads, d_ff, sigma,
                           top_k_values[i],
                           dropout,
                           grid_size=self.grid_size,
                           drop_path=dpr[i])
            for i in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        attention_weights = []
        attention_masks = []
        
        for block in self.blocks:
            x, attn_w, attn_mask = block(x)
            attention_weights.append(attn_w)
            attention_masks.append(attn_mask)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits, attention_weights, attention_masks


def get_mixup_cutmix_transforms(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=1000, mixup_prob=0.8):

    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        cutmix_transform = transforms_v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
        mixup_transform = transforms_v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
        
        def random_mixup_cutmix(inputs, targets):
            if np.random.rand() < mixup_prob:
                return cutmix_transform(inputs, targets)
            else:
                return mixup_transform(inputs, targets)
        
        return random_mixup_cutmix
    elif cutmix_alpha > 0.0:
        return transforms_v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
    elif mixup_alpha > 0.0:
        return transforms_v2.MixUp(num_classes=num_classes, alpha=mixup_alpha)
    else:
        return None


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

