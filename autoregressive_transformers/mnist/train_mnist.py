import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

IMAGE_H = 28
IMAGE_W = 28
IMAGE_C = 1
PIXELS_PER_ROW = IMAGE_W * IMAGE_C
SEQ_LEN = IMAGE_H * IMAGE_W * IMAGE_C

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(3407)

def get_2d_causal_mask(T, device):
    
    indices = torch.arange(T, device=device)
    
    q_i = (indices // PIXELS_PER_ROW).unsqueeze(1)
    q_j = ((indices % PIXELS_PER_ROW) // IMAGE_C).unsqueeze(1)
    q_c = (indices % IMAGE_C).unsqueeze(1)
    
    k_i = (indices // PIXELS_PER_ROW).unsqueeze(0)
    k_j = ((indices % PIXELS_PER_ROW) // IMAGE_C).unsqueeze(0)
    k_c = (indices % IMAGE_C).unsqueeze(0)
    
    mask_prev_row = (k_i < q_i)
    mask_prev_col = (k_i == q_i) & (k_j < q_j)
    mask_same_pixel_causal = (k_i == q_i) & (k_j == q_j) & (k_c <= q_c)
    mask_allowed = mask_prev_row | mask_prev_col | mask_same_pixel_causal
    mask_blocked = ~mask_allowed
    
    final_mask = mask_blocked.float().masked_fill(mask_blocked, float('-inf'))

    return final_mask.unsqueeze(0).unsqueeze(0) # (1, 1, T, T)

def get_window_mask(T, window_size, device):
    
    indices = torch.arange(T, device=device)
    q_indices = indices.unsqueeze(1)
    k_indices = indices.unsqueeze(0)
    
    mask_blocked = k_indices < q_indices - window_size + 1
    
    final_mask = torch.full((T, T), 0.0, device=device)
    final_mask = final_mask.masked_fill(mask_blocked, float('-inf'))

    return final_mask.unsqueeze(0).unsqueeze(0)

class KrauseAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, init_sigma=2.5, top_k=96, window_size=128, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.top_k = int(top_k)
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_sigma = nn.Parameter(torch.full((num_heads, 1, 1), math.log(init_sigma) / 1.0))

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.embed_dim,)
        return x.view(*new_shape)

    def forward(self, x):
        B, T, E = x.shape
        device = x.device
        
        q = self._split_heads(self.W_q(x)) # (B, H, T, D)
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))
        
        q_sq = torch.sum(q**2, dim=-1, keepdim=True)
        k_sq = torch.sum(k**2, dim=-1, keepdim=True).transpose(-2, -1)
        product = torch.matmul(q, k.transpose(-2, -1))
        dist_sq = q_sq + k_sq - 2 * product
        
        sigma_sq = torch.exp(2 * self.log_sigma) # (H, 1, 1)
        scores = dist_sq / (-2 * sigma_sq) # (B, H, T, T)
        
        mask_2d_causal = get_2d_causal_mask(T, device) # (1, 1, T, T)
        scores = scores + mask_2d_causal
        
        if self.window_size < T:
            mask_window = get_window_mask(T, self.window_size, device)
            scores = scores + mask_window

        if self.top_k < T:
            K_val = min(self.top_k, T)
            _, topk_indices = torch.topk(scores, k=K_val, dim=-1)
            
            inf_tensor = torch.full_like(scores, float('-inf'))
            topk_scores = torch.gather(scores, dim=-1, index=topk_indices)
            scores = inf_tensor.scatter_(dim=-1, index=topk_indices, src=topk_scores)
            
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        output = torch.matmul(probs, v)
        output = self._combine_heads(output)
        return self.out_proj(output)


class LinearAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, **kwargs):
        kwargs.pop('top_k', None)
        kwargs.pop('window_size', None)
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def feature_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.feature_map(q) # (B, H, T, D)
        k = self.feature_map(k) # (B, H, T, D)
        
        K_V_outer_product = torch.einsum('bhjd,bhje->bhjde', k, v)
        kv_cumsum = torch.cumsum(K_V_outer_product, dim=2)
        num = torch.matmul(q.unsqueeze(-2), kv_cumsum).squeeze(-2)
        
        k_cumsum = torch.cumsum(k, dim=2) # (B, H, T, D)
        den = (q * k_cumsum).sum(dim=-1).unsqueeze(-1) + self.eps
        
        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(out)


class VanillaAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, **kwargs):
        kwargs.pop('top_k', None)
        kwargs.pop('window_size', None)
        kwargs.pop('init_sigma', None)
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, E = x.shape
        device = x.device
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask_2d_causal = get_2d_causal_mask(T, device) # (1, 1, T, T)
        attn_scores = attn_scores + mask_2d_causal

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        output = self.out_proj(context.view(B, T, self.embed_dim))
        
        return output


class ImageTransformerBlock(nn.Module):

    def __init__(self, d_model, nhead, dropout, attn_type='vanilla', **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        
        if attn_type == 'krause':
            self.attn = KrauseAttention(d_model, nhead, dropout, **kwargs)
        elif attn_type == 'linear':
            self.attn = LinearAttention(d_model, nhead, dropout, **kwargs)
        else:
            self.attn = VanillaAttention(d_model, nhead, dropout, **kwargs)
            
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.res_dropout(self.attn(self.ln1(x)))
        x = x + self.res_dropout(self.mlp(self.ln2(x)))
        return x

class ImageTransformer(nn.Module):

    def __init__(self, vocab_size=256, seq_len=SEQ_LEN, d_model=256, nhead=8, num_layers=8, attn_type='vanilla', **kwargs):
        super().__init__()
        self.attn_type = attn_type
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            ImageTransformerBlock(d_model, nhead, 0.1, attn_type, **kwargs)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def setup(rank, world_size):

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def mnist_transform_func(x):
    return (x * 255).long().reshape(-1)

def get_mnist_loaders(batch_size, rank, world_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(mnist_transform_func)
    ])
    
    if rank == 0:
        datasets.MNIST('./data', train=True, download=True, transform=transform)
        datasets.MNIST('./data', train=False, download=True, transform=transform)
    dist.barrier()
    
    train_ds = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader, train_sampler

def compute_bpd(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_pixels = 0
    T = SEQ_LEN
    
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            x = imgs.view(imgs.size(0), -1)
            inp, target = x[:, :-1], x[:, 1:]

            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction='sum')
            
            total_loss += loss.item()
            total_pixels += target.numel()
    
    dist.all_reduce(torch.tensor([total_loss, total_pixels], device=device), op=dist.ReduceOp.SUM)
    
    avg_loss_per_pixel = total_loss / total_pixels
    bpd = avg_loss_per_pixel / math.log(2)
    return bpd

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * scale for base_lr in self.base_lrs]
        else:
            step_after_warmup = self.last_epoch - self.warmup_steps
            steps_in_cosine = self.total_steps - self.warmup_steps
            
            if steps_in_cosine <= 0:
                 return [self.min_lr for base_lr in self.base_lrs]
            
            scale = 0.5 * (1. + math.cos(math.pi * step_after_warmup / steps_in_cosine))
            return [self.min_lr + (self.max_lr - self.min_lr) * scale for base_lr in self.base_lrs]


def train_and_evaluate(args, attn_type, rank, world_size, train_loader, test_loader, train_sampler):
    device = torch.device(f'cuda:{rank}')
    is_rank_0 = (rank == 0)

    model_core = ImageTransformer(
        seq_len=SEQ_LEN,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        attn_type=attn_type, top_k=args.top_k, window_size=args.window_size
    ).to(device)
    model = DDP(model_core, device_ids=[device])
    
    if is_rank_0:
        print(f"\n[{attn_type.upper()}] Params: {sum(p.numel() for p in model_core.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        max_lr=args.lr, min_lr=args.lr * 1e-6
    )
    criterion = nn.CrossEntropyLoss()
    history = {'loss': [], 'bpd': [], 'time_per_epoch': []}
    total_steps_per_epoch = len(train_loader)
    log_interval = 50

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        t0 = time.time()

        if is_rank_0:
             print(f"--- Epoch {epoch+1}/{args.epochs} Start ---")
             
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            x = imgs.view(imgs.size(0), -1)
            inp, target = x[:, :-1], x[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits.reshape(-1, 256), target.reshape(-1))
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            dist_loss = loss.detach().clone()
            dist.all_reduce(dist_loss, op=dist.ReduceOp.SUM)
            avg_loss = dist_loss.item() / world_size
            epoch_loss += avg_loss

            if is_rank_0 and (i % log_interval == 0 or i == total_steps_per_epoch - 1):
                current_lr = scheduler.get_last_lr()[0]
                print(f" | Step {i+1}/{total_steps_per_epoch} | Loss (Avg): {avg_loss:.4f} | LR: {current_lr:.6f}")

        t1 = time.time()
        epoch_time = t1 - t0
        final_avg_loss = epoch_loss / len(train_loader)

        if is_rank_0:
            print(f" > Evaluating Epoch {epoch+1}...")
        dist.barrier()
        val_bpd = compute_bpd(model, test_loader, device)
        
        if is_rank_0:
            print(f"--- Epoch {epoch+1} Finished ---")
            print(f" | Avg Loss: {final_avg_loss:.4f} | Val BPD: {val_bpd:.4f} | Time: {epoch_time:.2f}s")
            history['loss'].append(final_avg_loss)
            history['bpd'].append(val_bpd)
            history['time_per_epoch'].append(epoch_time)

            bpd_len = len(history['bpd'])
            should_save = False
            if bpd_len == 1:
                should_save = True
            elif bpd_len > 1 and val_bpd < min(history['bpd'][:-1]):
                should_save = True
            
            if should_save:
                 model_dir = "saved_models_mnist"
                 os.makedirs(model_dir, exist_ok=True)
                 model_path = os.path.join(model_dir, f"mnist_{attn_type}_D{args.d_model}_L{args.num_layers}_K{args.top_k}_W{args.window_size}.pt")
                 
                 torch.save(model.module.state_dict(), model_path)
                 print(f" | Saved new BEST model to {model_path} (BPD: {val_bpd:.4f})")
        
    return model.module, history

def run_benchmark(rank, world_size, args):

    rank, local_rank, world_size = setup(rank, world_size)
    is_rank_0 = (rank == 0)

    if is_rank_0:
        print(f"{'='*50}")
        print(f"Starting DDP Benchmark on {world_size} GPUs.")
        print(f"Krause: Top-K={args.top_k}, Window={args.window_size}")
        print(f"{'='*50}")

    train_loader, test_loader, train_sampler = get_mnist_loaders(args.batch_size, rank, world_size)
    
    models = {}
    results = {}
    
    for attn_type in ['krause', 'linear', 'vanilla']:
        m, h = train_and_evaluate(args, attn_type, local_rank, world_size, train_loader, test_loader, train_sampler)
        if is_rank_0:
            results[attn_type], models[attn_type] = h, m
    
    if is_rank_0:
        print("\nBenchmark Complete. Plotting BPD history...")
        
        os.makedirs("benchmark_results_mnist", exist_ok=True)
        
        plt.figure(figsize=(6, 5))
        for name, res in results.items():
            plt.plot(res['bpd'], label=name)
        plt.title('Bits Per Dimension (BPD) during Training')
        plt.xlabel('Epoch')
        plt.ylabel('BPD')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("benchmark_results_mnist/bpd_history.png")
        print("Results saved to benchmark_results_mnist/")

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU, total batch size = batch_size * 4.')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--top_k', type=int, default=96)
    parser.add_argument('--window_size', type=int, default=128)
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Max learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Ratio of total steps for linear warmup.')
    args = parser.parse_args()
    
    if 'RANK' in os.environ:
        run_benchmark(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), args)
    else:
        world_size = torch.cuda.device_count()
        if world_size > 0:
             mp.spawn(run_benchmark, args=(world_size, args), nprocs=world_size, join=True)
        else:
            run_benchmark(0, 1, args)

if __name__ == '__main__':
    main()
