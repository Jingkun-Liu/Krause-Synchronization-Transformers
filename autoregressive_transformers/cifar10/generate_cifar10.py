import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms

IMAGE_H = 32
IMAGE_W = 32
IMAGE_C = 3
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

set_seed(20251220)

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
        raise NotImplementedError("Generation uses forward_step")

    @torch.no_grad()
    def forward_step(self, x, state=None, current_pos=0):
        B, T_curr, E = x.shape
        device = x.device
        
        q = self._split_heads(self.W_q(x)) # (B, H, 1, D_h)
        k = self._split_heads(self.W_k(x)) # (B, H, 1, D  _h)
        v = self._split_heads(self.W_v(x)) # (B, H, 1, D_h)
        
        W = self.window_size
        
        if state is None:
            K_ring = torch.zeros(B, self.num_heads, W, self.head_dim, device=device)
            V_ring = torch.zeros(B, self.num_heads, W, self.head_dim, device=device)
        else:
            K_ring, V_ring = state
        
        idx = current_pos % W
        
        K_ring[:, :, idx, :] = k.squeeze(2)
        V_ring[:, :, idx, :] = v.squeeze(2)

        current_len = min(current_pos + 1, W)
        
        if current_pos < W:
            K_window = K_ring[:, :, :current_len]
            V_window = V_ring[:, :, :current_len]
        else:
            start_idx = (current_pos + 1) % W
            K_window = torch.cat([K_ring[:, :, start_idx:], K_ring[:, :, :start_idx]], dim=2)
            V_window = torch.cat([V_ring[:, :, start_idx:], V_ring[:, :, :start_idx]], dim=2)
        
        q_expanded = q.unsqueeze(3)
        k_window_expanded = K_window.unsqueeze(2)
        
        diff = q_expanded - k_window_expanded
        dist_sq = torch.sum(diff**2, dim=-1).squeeze(2)

        sigma_sq = torch.exp(2 * self.log_sigma).unsqueeze(0)
        raw_scores = dist_sq / (-2 * sigma_sq.squeeze(-1))

        actual_top_k = min(self.top_k, raw_scores.size(-1))
        scores_top_k, indices_top_k = torch.topk(raw_scores, k=actual_top_k, dim=-1)
        probs_top_k = F.softmax(scores_top_k, dim=-1)
        
        idx_expanded = indices_top_k.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        v_top_k = torch.gather(V_window, 2, idx_expanded)
        
        consensus = torch.einsum('bhk, bhkd -> bhd', probs_top_k, v_top_k)
        consensus = self._combine_heads(consensus.unsqueeze(2))
        output = self.out_proj(consensus)
        
        next_state = (K_ring, V_ring)
        return output, next_state

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
        raise NotImplementedError("Generation uses forward_step")

    @torch.no_grad()
    def forward_step(self, x, state=None):
        B, _, C = x.shape
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.feature_map(q)
        k = self.feature_map(k)

        if state is None:
            kv_state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
            k_state = torch.zeros(B, self.num_heads, self.head_dim, device=x.device)
        else:
            kv_state, k_state = state

        kv_state = kv_state + torch.einsum('bhtd,bhte->bhde', k, v)
        k_state = k_state + k.squeeze(2)

        num = torch.einsum('bhtd,bhde->bhte', q, kv_state)
        den = torch.einsum('bhtd,bhd->bht', q, k_state).unsqueeze(-1) + self.eps
        
        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, 1, C)
        
        return self.out_proj(out), (kv_state, k_state)

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

    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.embed_dim,)
        return x.view(*new_shape)

    def forward(self, x):
        raise NotImplementedError("Generation uses forward_step")

    @torch.no_grad()
    def forward_step(self, x, state=None, current_pos=0):
        B, T_curr, E = x.shape
        device = x.device
        
        q = self._split_heads(self.q_proj(x)) # (B, H, 1, D_h)
        k = self._split_heads(self.k_proj(x)) # (B, H, 1, D_h)
        v = self._split_heads(self.v_proj(x)) # (B, H, 1, D_h)

        if state is None:
            K_cache = k
            V_cache = v
        else:
            K_prev, V_prev = state
            K_cache = torch.cat([K_prev, k], dim=2)
            V_cache = torch.cat([V_prev, v], dim=2)

        attn_scores = torch.matmul(q, K_cache.transpose(-1, -2))
        
        attn_weights = F.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1) # (B, H, 1, T_full)
        
        context = torch.matmul(attn_weights, V_cache)
        
        output = self._combine_heads(context)
        output = self.out_proj(output)

        next_state = (K_cache, V_cache)
        return output, next_state

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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    @torch.no_grad()
    def forward_step(self, x, state=None, current_pos=0):
        x_ln = self.ln1(x)
        
        if isinstance(self.attn, KrauseAttention):
            attn_out, next_state = self.attn.forward_step(x_ln, state, current_pos=current_pos)
        elif isinstance(self.attn, LinearAttention):
            attn_out, next_state = self.attn.forward_step(x_ln, state)
        elif isinstance(self.attn, VanillaAttention):
            attn_out, next_state = self.attn.forward_step(x_ln, state)
        else:
            raise NotImplementedError("Incremental stepping only supported for Linear, Krause, and Vanilla.")
            
        x = x + self.res_dropout(attn_out)
        x = x + self.res_dropout(self.mlp(self.ln2(x)))
        return x, next_state

class ImageTransformer(nn.Module):
    def __init__(self, vocab_size=256, seq_len=3072, d_model=512, nhead=8,
                 num_layers=8, attn_type='vanilla', **kwargs):
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
        return self.head(h)

    @torch.no_grad()
    def generate(self, device, num_samples, gen_batch_size, is_ddp=False):
        self.eval()
        
        all_generated_pixels = []
        num_batches = (num_samples + gen_batch_size - 1) // gen_batch_size
        
        print(f"Generating {num_samples} images with {self.attn_type} on {device}...")
        print(f"Using batch size: {gen_batch_size}, total batches: {num_batches}")
        
        t0 = time.time()
        
        for batch_idx in tqdm(range(num_batches), desc=f"Batch Gen ({self.attn_type})"):
            
            current_batch_size = min(gen_batch_size, num_samples - batch_idx * gen_batch_size)
            if current_batch_size <= 0:
                break
                
            generated = torch.randint(0, 256, (current_batch_size, 1), dtype=torch.long, device=device)
            
            states = [None] * len(self.layers)
            
            for t in tqdm(range(self.seq_len - 1), desc="Generating"):
                current_token = generated[:, -1:].to(device)
                
                tok_emb = self.token_emb(current_token)
                pos_emb = self.pos_emb(torch.tensor([t], device=device).long())
                h = self.drop(tok_emb + pos_emb) # [B, 1, E]
                
                new_states = []
                for i, layer in enumerate(self.layers):
                    h, s = layer.forward_step(h, states[i], current_pos=t)
                    new_states.append(s)
                states = new_states
                
                h = self.ln_f(h)
                logits = self.head(h)
                
                probs = F.softmax(logits[:, -1, :] / 1.0, dim=-1)
                next_pixel = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat((generated, next_pixel), dim=1)
            
            all_generated_pixels.append(generated.cpu())

        total_time = time.time() - t0
        
        final_pixels = torch.cat(all_generated_pixels, dim=0)
        generated_images_np = final_pixels.view(-1, IMAGE_H, IMAGE_W, IMAGE_C).cpu().numpy().astype(np.uint8)

        images_per_sec = num_samples / total_time
        pixels_per_sec = (num_samples * self.seq_len) / total_time
        
        print(f"\n Generation finished for {self.attn_type}. Speed: {pixels_per_sec:.4f} pixels/sec | {images_per_sec:.4f} images/sec")
        
        return generated_images_np

def generate_only_and_plot():
    parser = argparse.ArgumentParser(description="Image Transformer Generation from Saved Checkpoints.")
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (D).')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of transformer layers (L).')
    parser.add_argument('--top_k', type=int, default=192, help='Krause Attention Top-K parameter.')
    parser.add_argument('--window_size', type=int, default=256, help='Krause Attention Window Size parameter.')
    parser.add_argument('--generate_samples', type=int, default=10000, help='Number of images to generate per model.')
    parser.add_argument('--gen_batch_size', type=int, default=200, help='Batch size for generation to control memory usage.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    print(f"\n{'='*50}")
    print(f"CIFAR-10 Transformer Generation | Device: {device}")
    print(f"Params: D={args.d_model}, L={args.num_layers}")
    print(f"Generation Batch Size: {args.gen_batch_size}")
    print(f"Generation Samples: {args.generate_samples}")
    print(f"{'='*50}")

    attn_types = ['krause', 'linear', 'vanilla']
    models_and_samples = {}
    
    for attn_type in attn_types:
            
        model_dir = "saved_models_cifar10"
        model_path = os.path.join(model_dir, f"cifar10_{attn_type}_D{args.d_model}_L{args.num_layers}_K{args.top_k}_W{args.window_size}.pt")
        
        model_core = ImageTransformer(
            seq_len=3072,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            attn_type=attn_type, top_k=args.top_k, window_size=args.window_size
        ).to(device)
        
        print(f"\n--- Processing {attn_type.upper()} ---")

        state_dict = torch.load(model_path, map_location='cpu')
        model_core.load_state_dict(state_dict)
        model_core = model_core.to(device)
        
        generated_images_np = model_core.generate(device, num_samples=args.generate_samples, gen_batch_size=args.gen_batch_size)

        models_and_samples[attn_type] = torch.from_numpy(generated_images_np).cpu()

    if models_and_samples:
        print("\nAll models finished. Plotting...")
        os.makedirs("generation_results_cifar10", exist_ok=True)

        num_rows = 3
        num_cols = 8
        total_samples = num_rows * num_cols
        
        samples = models_and_samples['krause'].numpy()
        plot_samples = samples[:total_samples]
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4.5))
        
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx < len(plot_samples):
                    ax = axes[row, col]
                    ax.imshow(plot_samples[idx].astype(np.uint8))
                    ax.axis('off')
        
        plt.tight_layout()
        plot_path = "generation_results_cifar10/generated_samples_cifar10.png"
        plt.savefig(plot_path)

if __name__ == "__main__":
    generate_only_and_plot()