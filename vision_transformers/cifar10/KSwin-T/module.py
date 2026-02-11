import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision.transforms.v2 as transforms_v2
from torchvision.models.swin_transformer import SwinTransformer, PatchMerging


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
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class KrauseWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size=0, sigma=1.0,
                 qkv_bias=True, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        
        if isinstance(shift_size, int):
            shift_size = (shift_size, shift_size)
        elif isinstance(shift_size, (list, tuple)):
            shift_size = tuple(shift_size)
        else:
            shift_size = (0, 0)
        self.shift_size = shift_size
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        
        self.log_sigma = nn.Parameter(torch.ones(num_heads) * math.log(sigma))
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def _get_relative_position_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()
    
    def rbf_attention_weights(self, query, key):
        B_, heads, N, D = query.shape
        sigma_sq = torch.exp(2 * self.log_sigma).view(1, heads, 1, 1)
        
        query = query * self.scale
        key = key * self.scale
        
        q_sq = (query ** 2).sum(-1, keepdim=True)
        k_sq = (key ** 2).sum(-1, keepdim=True).transpose(-2, -1)
        qk = torch.matmul(query, key.transpose(-2, -1))
        dist_sq = q_sq + k_sq - 2 * qk
        dist_sq = torch.clamp(dist_sq, min=1e-12)
        
        scores = -0.5 / sigma_sq * dist_sq
        return scores
    
    def forward(self, x):
        B, H, W, C = x.shape
        win_h, win_w = self.window_size
        shift_h, shift_w = self.shift_size
        N = min(win_h * win_w, H * W)
        
        attn_mask = None
        shifted_x = x
        if (shift_h > 0 or shift_w > 0) and H >= win_h and W >= win_w:
            shifted_x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -win_h),
                        slice(-win_h, -shift_h),
                        slice(-shift_h, None))
            w_slices = (slice(0, -win_w),
                        slice(-win_w, -shift_w),
                        slice(-shift_w, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            if H >= win_h and W >= win_w:
                mask_windows = self._window_partition(img_mask.view(1, 1, H, W, 1))
                mask_windows = mask_windows.view(-1, N)
            else:
                mask_windows = None
            
            if mask_windows is not None:
                attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
                attn_mask = attn_mask.to(x.dtype)
            else:
                attn_mask = None
        
        qkv = self.qkv(shifted_x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)
        
        q = self._window_partition(q)
        k = self._window_partition(k)
        v = self._window_partition(v)
        
        B_, heads, win_h_actual, win_w_actual, head_dim = q.shape
        N_actual = win_h_actual * win_w_actual
        q = q.reshape(B_, heads, N_actual, head_dim)
        k = k.reshape(B_, heads, N_actual, head_dim)
        v = v.reshape(B_, heads, N_actual, head_dim)
        
        attn = self.rbf_attention_weights(q, k)
        
        if win_h_actual == self.window_size[0] and win_w_actual == self.window_size[1]:
            attn = attn + self._get_relative_position_bias().unsqueeze(0)
        
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B, nW, heads, N_actual, N_actual) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, heads, N_actual, N_actual)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, win_h_actual, win_w_actual, C)
        
        x = self._window_reverse(x, H, W)
        
        if (shift_h > 0 or shift_w > 0) and H >= win_h and W >= win_w:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _window_partition(self, x):
        B, heads, H, W, head_dim = x.shape
        win_h, win_w = self.window_size
        
        if H < win_h or W < win_w:
            windows = x.view(B, heads, H, W, head_dim)
            return windows.view(-1, heads, H, W, head_dim)
        
        x = x.view(B, heads, H // win_h, win_h, W // win_w, win_w, head_dim)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        windows = windows.view(-1, heads, win_h, win_w, head_dim)
        return windows
    
    def _window_reverse(self, windows, H, W):
        win_h, win_w = self.window_size

        if H < win_h or W < win_w:
            return windows
        
        num_win_h = H // win_h
        num_win_w = W // win_w
        B_nW, _, _, C = windows.shape
        B = B_nW // (num_win_h * num_win_w)
        x = windows.view(B, num_win_h, num_win_w, win_h, win_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        return x

def create_krause_swin_t(num_classes=10, sigma=1.0, drop_path=0.0):
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = 4
    mlp_ratio = 4.0
    dropout = 0.0
    
    total_blocks = sum(depths)
    dpr = [x.item() for x in torch.linspace(0, drop_path, total_blocks)] if total_blocks > 1 else [drop_path]
    
    base_model = SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=(window_size, window_size),
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=dropout,
        stochastic_depth_prob=drop_path,
        num_classes=num_classes,
        norm_layer=nn.LayerNorm,
        block=None, 
        downsample_layer=PatchMerging,
    )
    
    def replace_attention(module, sigma):
        for name, child in list(module.named_children()):
            class_name = child.__class__.__name__
            if class_name in ('WindowAttention', 'ShiftedWindowAttention'):
                dim = child.qkv.in_features
                num_heads = child.num_heads
                window_size = getattr(child, 'window_size', (4, 4))
                shift_size = getattr(child, 'shift_size', (0, 0))
                
                if isinstance(window_size, int):
                    window_size = (window_size, window_size)
                if isinstance(shift_size, int):
                    shift_size = (shift_size, shift_size)
                
                new_attn = KrauseWindowAttention(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift_size=shift_size,
                    sigma=sigma,
                    qkv_bias=child.qkv.bias is not None,
                    proj_bias=child.proj.bias is not None,
                    attn_drop=child.attn_drop.p if hasattr(child, 'attn_drop') else 0.0,
                    proj_drop=child.proj_drop.p if hasattr(child, 'proj_drop') else 0.0,
                )
                
                with torch.no_grad():
                    new_attn.qkv.weight.copy_(child.qkv.weight)
                    if child.qkv.bias is not None:
                        new_attn.qkv.bias.copy_(child.qkv.bias)
                    new_attn.proj.weight.copy_(child.proj.weight)
                    if child.proj.bias is not None:
                        new_attn.proj.bias.copy_(child.proj.bias)
                    if hasattr(child, 'relative_position_bias_table'):
                        new_attn.relative_position_bias_table.copy_(child.relative_position_bias_table)
                    if hasattr(child, 'relative_position_index'):
                        new_attn.register_buffer('relative_position_index', child.relative_position_index.clone())
                
                setattr(module, name, new_attn)
                print(f"Replaced {class_name} (shift_size={shift_size}) with KrauseWindowAttention")
            else:
                replace_attention(child, sigma)
    
    print("\nReplacing attention modules in SwinTransformer...")
    
    replace_attention(base_model, sigma)
    
    print("All attention modules replaced successfully\n")
    
    return base_model

def get_mixup_cutmix_transforms(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=10, mixup_prob=0.8):
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


def get_all_sigma_values(model):
    sigma_values = []
    
    def collect_sigma(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, KrauseWindowAttention):
                current_sigma = torch.exp(child.log_sigma).mean().item()
                sigma_values.append((full_name, current_sigma))
            else:
                collect_sigma(child, full_name)
    
    collect_sigma(model)

    return sigma_values
