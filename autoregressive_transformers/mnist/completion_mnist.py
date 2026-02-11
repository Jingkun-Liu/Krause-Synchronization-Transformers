import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

IMAGE_H = 28
IMAGE_W = 28
IMAGE_C = 1
PIXELS_PER_ROW = IMAGE_W * IMAGE_C

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(9999)


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
    return final_mask.unsqueeze(0).unsqueeze(0)

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
        
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))
        
        q_sq = torch.sum(q**2, dim=-1, keepdim=True)
        k_sq = torch.sum(k**2, dim=-1, keepdim=True).transpose(-2, -1)
        product = torch.matmul(q, k.transpose(-2, -1))
        dist_sq = q_sq + k_sq - 2 * product
        
        sigma_sq = torch.exp(2 * self.log_sigma)
        scores = dist_sq / (-2 * sigma_sq)
        
        scores = scores + get_2d_causal_mask(T, device)
        if self.window_size < T:
            scores = scores + get_window_mask(T, self.window_size, device)
        
        # Top-K
        if self.top_k < T:
            K_val = min(self.top_k, T)
            _, topk_indices = torch.topk(scores, k=K_val, dim=-1)
            inf_tensor = torch.full_like(scores, float('-inf'))
            topk_scores = torch.gather(scores, dim=-1, index=topk_indices)
            scores = inf_tensor.scatter_(dim=-1, index=topk_indices, src=topk_scores)
            
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        output = torch.matmul(probs, v)
        return self.out_proj(self._combine_heads(output))

class LinearAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.eps = 1e-6

    def feature_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.feature_map(q), self.feature_map(k)
        
        kv_cumsum = torch.cumsum(torch.einsum('bhjd,bhje->bhjde', k, v), dim=2)
        num = torch.matmul(q.unsqueeze(-2), kv_cumsum).squeeze(-2)
        den = (q * torch.cumsum(k, dim=2)).sum(dim=-1).unsqueeze(-1) + self.eps
        
        return self.out_proj((num/den).transpose(1, 2).contiguous().view(B, T, E))

class VanillaAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, E = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + get_2d_causal_mask(T, x.device).squeeze(0).squeeze(0)
        
        probs = self.attn_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(probs, V)
        return self.out_proj(context.transpose(1, 2).contiguous().view(B, T, E))

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
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.res_dropout(self.attn(self.ln1(x)))
        x = x + self.res_dropout(self.mlp(self.ln2(x)))
        return x

class ImageTransformer(nn.Module):

    def __init__(self, vocab_size=256, seq_len=784, d_model=256, nhead=8, num_layers=8, attn_type='vanilla', **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([ImageTransformerBlock(d_model, nhead, 0.1, attn_type, **kwargs) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(positions))
        for layer in self.layers: h = layer(h)
        return self.head(self.ln_f(h))

def mnist_transform_func(x):
    return (x * 255).long().reshape(-1)

def get_test_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(mnist_transform_func)])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

def load_model_from_file(path, args, device):
    print(f"Initializing model: {args.attn_type}, D={args.d_model}, L={args.num_layers}, K={args.top_k}, W={args.window_size}")
    model = ImageTransformer(
        vocab_size=256,
        seq_len=784,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        attn_type=args.attn_type,
        top_k=args.top_k,
        window_size=args.window_size
    ).to(device)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading state dict from: {path}")
    state_dict = torch.load(path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

@torch.no_grad()
def generate_completion(model, prompt, total_len, temperature=1.0):
    model.eval()
    curr_seq = prompt.clone()
    
    print(f"Generating from token {curr_seq.shape[1]} to {total_len}...")
    
    for _ in range(curr_seq.shape[1], total_len):
        logits = model(curr_seq)
        next_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        curr_seq = torch.cat([curr_seq, next_token], dim=1)
        
    return curr_seq

def sequences_to_images(seqs):
    return seqs.reshape(seqs.shape[0], IMAGE_H, IMAGE_W, IMAGE_C).cpu().numpy().astype(np.uint8).squeeze(-1)

def plot_completions(original, completed, samples_per_img, filename):
    num_imgs = len(original)
    cols = samples_per_img + 2
    
    fig, axes = plt.subplots(num_imgs, cols, figsize=(2*cols, 2*num_imgs))
    if num_imgs == 1: axes = axes[None, :]
        
    for i in range(num_imgs):
        masked = original[i].copy()
        masked[14:, :] = 255
        
        axes[i, 0].imshow(masked, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title("Input (Half)")
        axes[i, 0].axis('off')

        for j in range(samples_per_img):
            axes[i, j+1].imshow(completed[i*samples_per_img + j], cmap='gray', vmin=0, vmax=255)
            axes[i, j+1].set_title(f"Sample {j+1}")
            axes[i, j+1].axis('off')

        axes[i, -1].imshow(original[i], cmap='gray', vmin=0, vmax=255)
        axes[i, -1].set_title("Ground Truth")
        axes[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved result to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saved_models_mnist/mnist_krause_D256_L12_K96_W128.pt', help='Path to the trained model checkpoint.')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--attn_type', type=str, default='krause', choices=['krause', 'linear', 'vanilla'])
    parser.add_argument('--top_k', type=int, default=96)
    parser.add_argument('--window_size', type=int, default=128)
    
    parser.add_argument('--num_test_images', type=int, default=6, help='How many source images to complete')
    parser.add_argument('--samples_per_image', type=int, default=6, help='How many variations per image')
    parser.add_argument('--temp', type=float, default=1.0)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model_from_file(args.model_path, args, device)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(mnist_transform_func)])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    num_images = args.num_test_images
    last_indices = list(range(len(test_ds) - num_images, len(test_ds)))
    imgs = torch.stack([test_ds[i][0] for i in last_indices])
    imgs = imgs.to(device)
    
    half_len = 14 * 28
    total_len = 28 * 28
    
    prompt = imgs[:, :half_len]
    prompt_repeated = prompt.repeat_interleave(args.samples_per_image, dim=0)
    
    print(f"Starting completion for {args.num_test_images} images with {args.samples_per_image} samples each...")
    completed_seqs = generate_completion(model, prompt_repeated, total_len, args.temp)
    
    orig_np = sequences_to_images(imgs)
    comp_np = sequences_to_images(completed_seqs)
    
    out_file = f"completion_mnist_{args.attn_type}_D{args.d_model}_L{args.num_layers}.png"
    plot_completions(orig_np, comp_np, args.samples_per_image, out_file)

if __name__ == '__main__':
    main()
