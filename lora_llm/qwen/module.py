import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    Qwen2Config,
    repeat_kv,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)


class KrauseQwenAttention(nn.Module):

    def __init__(self, config: Qwen2Config, params: dict, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Krause Params
        self.window_size = params.get("window_size", 32)
        self.top_k = params.get("top_k", 16)
        init_sigma = params.get("init_sigma", 5.5)
        self.log_sigma = nn.Parameter(torch.full((self.num_heads, 1, 1), math.log(init_sigma)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_proj_krause = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_krause = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_krause = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_krause = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.gate_proj_krause = nn.Linear(self.hidden_size * 2, 2, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.gate_proj_krause.weight, mean=0.0, std=0.01)
            if self.gate_proj_krause.bias is not None:
                init_logit_diff = math.log(0.8 / 0.2)
                self.gate_proj_krause.bias.data[0] = init_logit_diff
                self.gate_proj_krause.bias.data[1] = 0.0

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output_v, attn_weights_v = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output_v = attn_output_v.reshape(*input_shape, -1).contiguous()
        output_v = self.o_proj(attn_output_v)

        B, T, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q_k = self.q_proj_krause(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_k = self.k_proj_krause(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_k = self.v_proj_krause(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q_k_rot, k_k_rot = apply_rotary_pos_emb(q_k, k_k, cos, sin)

        k_k_full = repeat_kv(k_k_rot, self.num_key_value_groups)
        v_k_full = repeat_kv(v_k, self.num_key_value_groups)

        q_sq = torch.sum(q_k_rot.to(torch.float32) ** 2, dim=-1, keepdim=True)
        k_sq = torch.sum(k_k_full.to(torch.float32) ** 2, dim=-1, keepdim=True).transpose(-2, -1)
        dist = q_sq + k_sq - 2 * torch.matmul(
            q_k_rot.to(torch.float32), k_k_full.to(torch.float32).transpose(-2, -1)
        )

        sigma = torch.exp(self.log_sigma).to(torch.float32)
        scores_k = (dist / (-2 * sigma**2)).to(dtype)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k_k_full.shape[-2]]
            scores_k = scores_k + causal_mask

        neg_inf = torch.finfo(scores_k.dtype).min

        if self.window_size < T:
            q_idx = torch.arange(T, device=device).view(-1, 1)
            k_idx = torch.arange(T, device=device).view(1, -1)
            window_mask = k_idx < q_idx - self.window_size + 1
            scores_k = scores_k.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), neg_inf)

        if self.top_k < T:
            top_k_vals, _ = torch.topk(scores_k, k=min(self.top_k, T), dim=-1)
            min_val = top_k_vals[..., -1].unsqueeze(-1)
            scores_k = torch.where(
                scores_k < min_val,
                torch.tensor(neg_inf, dtype=scores_k.dtype, device=device),
                scores_k,
            )

        attn_weights_k = F.softmax(scores_k, dim=-1, dtype=torch.float32).to(dtype)
        attn_weights_k = F.dropout(
            attn_weights_k,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        attn_output_k = torch.matmul(attn_weights_k, v_k_full)
        attn_output_k = attn_output_k.transpose(1, 2).contiguous().reshape(B, T, -1)
        output_k = self.o_proj_krause(attn_output_k)

        if not self.training and kwargs.get("output_attentions", False):
            self.last_attn_weights = attn_weights_k.detach().cpu()

        concat_features = torch.cat([output_v, output_k], dim=-1)
        gate_logits = self.gate_proj_krause(concat_features)
        gate_weights = F.softmax(gate_logits, dim=-1)

        weight_v = gate_weights[:, :, 0:1]
        weight_k = gate_weights[:, :, 1:2]
        output = weight_v * output_v + weight_k * output_k

        return output, attn_weights_v
