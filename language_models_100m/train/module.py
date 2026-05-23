import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

class KrauseStandardMixAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float,
        window_size: int,
        top_k: int,
        init_sigma: float,
        init_standard_weight: float = 0.8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.top_k = top_k

        # Standard branch.
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Krause branch.
        self.q_proj_krause = nn.Linear(hidden_size, hidden_size)
        self.k_proj_krause = nn.Linear(hidden_size, hidden_size)
        self.v_proj_krause = nn.Linear(hidden_size, hidden_size)
        self.o_proj_krause = nn.Linear(hidden_size, hidden_size)
        self.log_sigma = nn.Parameter(torch.full((num_heads, 1, 1), math.log(init_sigma)))

        self.gate_proj_krause = nn.Linear(hidden_size * 2, 2, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.gate_proj_krause.weight, mean=0.0, std=0.01)
            if self.gate_proj_krause.bias is not None:
                init_standard_weight = min(max(init_standard_weight, 1e-4), 1 - 1e-4)
                init_logit_diff = math.log(init_standard_weight / (1.0 - init_standard_weight))
                self.gate_proj_krause.bias.data[0] = init_logit_diff
                self.gate_proj_krause.bias.data[1] = 0.0

        self.attn_dropout = nn.Dropout(attn_dropout)

    def _standard_branch(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)

    def _krause_branch(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj_krause(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj_krause(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj_krause(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        q_sq = torch.sum(qf**2, dim=-1, keepdim=True)
        k_sq = torch.sum(kf**2, dim=-1, keepdim=True).transpose(-2, -1)
        dist = q_sq + k_sq - 2 * torch.matmul(qf, kf.transpose(-2, -1))

        sigma = torch.exp(self.log_sigma).to(torch.float32)
        scores = (-dist / (2 * sigma**2)).to(x.dtype)

        causal_mask = build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

        if self.window_size > 0 and self.window_size < seq_len:
            q_idx = torch.arange(seq_len, device=x.device).view(-1, 1)
            k_idx = torch.arange(seq_len, device=x.device).view(1, -1)
            old_mask = (q_idx - k_idx) >= self.window_size
            scores = scores.masked_fill(old_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

        if self.top_k > 0 and self.top_k < seq_len:
            topk_vals, _ = torch.topk(scores, k=self.top_k, dim=-1)
            min_topk = topk_vals[..., -1].unsqueeze(-1)
            scores = torch.where(
                scores < min_topk,
                torch.full_like(scores, torch.finfo(scores.dtype).min),
                scores,
            )

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj_krause(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_v = self._standard_branch(x)
        output_k = self._krause_branch(x)

        concat_features = torch.cat([output_v, output_k], dim=-1)
        gate_logits = self.gate_proj_krause(concat_features)
        gate_weights = F.softmax(gate_logits, dim=-1)

        weight_v = gate_weights[:, :, 0:1]
        weight_k = gate_weights[:, :, 1:2]
        return weight_v * output_v + weight_k * output_k


def build_krause_attention(
    hidden_size: int,
    num_heads: int,
    attn_dropout: float,
    window_size: int,
    top_k: int,
    init_sigma: float,
    init_standard_weight: float = 0.8,
) -> nn.Module:
    return KrauseStandardMixAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        window_size=window_size,
        top_k=top_k,
        init_sigma=init_sigma,
        init_standard_weight=init_standard_weight,
    )
