import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types
import torchvision.transforms.v2 as transforms_v2
from torchvision.models.vision_transformer import VisionTransformer


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


def create_vit_model(img_size, patch_size, d_model, n_heads, n_layers, d_ff, num_classes, dropout):
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=n_layers,
        num_heads=n_heads,
        hidden_dim=d_model,
        mlp_dim=d_ff,
        num_classes=num_classes,
        dropout=dropout
    )
    
    model.conv_proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
    model.hidden_dim = d_model
    
    num_patches = (img_size // patch_size) ** 2
    model.encoder.pos_embedding = nn.Parameter(
        torch.randn(1, num_patches + 1, d_model)
    )
    model.encoder.seq_length = num_patches + 1

    def custom_process_input(self, x):
        n, c, h, w = x.shape
        p = patch_size
        torch._assert(h == img_size, f"Wrong image height! Expected {img_size} but got {h}!")
        torch._assert(w == img_size, f"Wrong image width! Expected {img_size} but got {w}!")
        n_h = h // p
        n_w = w // p
                
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        
        x = x.permute(0, 2, 1)
        
        return x
    
    model._process_input = types.MethodType(custom_process_input, model)
    
    return model
