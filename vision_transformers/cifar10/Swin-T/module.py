import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.v2 as transforms_v2


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
