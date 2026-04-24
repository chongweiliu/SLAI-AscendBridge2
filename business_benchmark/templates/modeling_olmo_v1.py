"""
Lightweight OLMo v1 model implementation for legacy HF checkpoints.

This helper avoids pulling the full ai2-olmo runtime during phase-4 business
benchmark runs. It is intentionally minimal and only implements the loading and
generation path needed by these legacy checkpoints.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput


class OlmoV1Config(PretrainedConfig):
    model_type = "olmo"

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        mlp_ratio: int = 4,
        vocab_size: int = 50304,
        embedding_size: int = 50304,
        max_sequence_length: int = 2048,
        weight_tying: bool = True,
        rope: bool = True,
        rope_full_precision: bool = True,
        swiglu: bool = True,
        activation_type: str = "swiglu",
        attention_layer_norm: bool = True,
        attention_layer_norm_with_affine: bool = True,
        include_bias: bool = True,
        residual_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        layer_norm_type: str = "low_precision",
        layer_norm_with_affine: bool = True,
        flash_attention: bool = False,
        scale_logits: bool = False,
        bias_for_layer_norm: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.weight_tying = weight_tying
        self.rope = rope
        self.rope_full_precision = rope_full_precision
        self.swiglu = swiglu
        self.activation_type = activation_type
        self.attention_layer_norm = attention_layer_norm
        self.attention_layer_norm_with_affine = attention_layer_norm_with_affine
        self.include_bias = include_bias
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.embedding_dropout = embedding_dropout
        self.layer_norm_type = layer_norm_type
        self.layer_norm_with_affine = layer_norm_with_affine
        self.flash_attention = flash_attention
        self.scale_logits = scale_logits
        self.bias_for_layer_norm = bias_for_layer_norm
        self.head_dim = d_model // n_heads


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class OlmoRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, full_precision: bool = True):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.full_precision = full_precision

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.full_precision:
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        return emb.cos(), emb.sin()


class OlmoBlock(nn.Module):
    def __init__(self, config: OlmoV1Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        mlp_hidden = config.mlp_ratio * config.d_model
        self.att_proj = nn.Linear(config.d_model, 3 * config.n_heads * head_dim, bias=config.include_bias)
        self.attn_norm = nn.LayerNorm(config.d_model, elementwise_affine=config.attention_layer_norm_with_affine) if config.attention_layer_norm else nn.Identity()
        self.attn_out = nn.Linear(config.n_heads * head_dim, config.d_model, bias=config.include_bias)
        self.q_norm = nn.LayerNorm(config.d_model, elementwise_affine=config.attention_layer_norm_with_affine)
        self.k_norm = nn.LayerNorm(config.d_model, elementwise_affine=config.attention_layer_norm_with_affine)
        self.rope = OlmoRotaryEmbedding(head_dim, config.max_sequence_length, full_precision=config.rope_full_precision)
        self.ff_proj = nn.Linear(config.d_model, mlp_hidden, bias=config.include_bias)
        self.ff_norm = nn.LayerNorm(config.d_model, elementwise_affine=config.layer_norm_with_affine)
        self.ff_out = nn.Linear(mlp_hidden // 2, config.d_model, bias=config.include_bias)
        self.residual_dropout = nn.Dropout(config.residual_dropout) if config.residual_dropout > 0 else nn.Identity()
        self.attention_dropout = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        del position_ids
        batch_size, seq_len, hidden_size = x.shape
        head_dim = 2 * self.rope.inv_freq.shape[0]
        n_heads = self.att_proj.out_features // (3 * head_dim)

        x_norm = self.attn_norm(x)
        qkv = self.att_proj(x_norm).reshape(batch_size, seq_len, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)

        cos, sin = self.rope(q, seq_len=seq_len)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_rope = (q * cos) + (_rotate_half(q) * sin)
        k_rope = (k * cos) + (_rotate_half(k) * sin)
        q = self.q_norm(q_rope.reshape(batch_size, seq_len, hidden_size)).reshape(batch_size, seq_len, n_heads, head_dim)
        k = self.k_norm(k_rope.reshape(batch_size, seq_len, hidden_size)).reshape(batch_size, seq_len, n_heads, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        kv_len = k.shape[2]
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=kv_len - seq_len,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        x = self.residual_dropout(x) + self.attn_out(attn_output)

        x_mlp = self.ff_norm(x)
        gate_up = self.ff_proj(x_mlp)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.residual_dropout(x) + self.ff_out(F.silu(gate) * up)

        if use_cache:
            return x, (k, v)
        return x, None


class OlmoTransformer(nn.Module):
    def __init__(self, config: OlmoV1Config):
        super().__init__()
        self.wte = nn.Embedding(config.embedding_size, config.d_model)
        self.blocks = nn.ModuleList([OlmoBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, elementwise_affine=config.layer_norm_with_affine)
        self.ff_out = nn.Linear(config.d_model, config.embedding_size, bias=False)
        self.dropout = nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0 else nn.Identity()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if input_ids is None:
            raise ValueError("input_ids is required")
        x = self.dropout(self.wte(input_ids))
        present_key_values = [] if use_cache else None
        for index, block in enumerate(self.blocks):
            past_kv = past_key_values[index] if past_key_values is not None else None
            x, present_kv = block(x, position_ids=position_ids, past_key_value=past_kv, use_cache=use_cache)
            if present_key_values is not None:
                present_key_values.append(present_kv)
        x = self.ln_f(x)
        return x, present_key_values


class OlmoV1Model(PreTrainedModel):
    config_class = OlmoV1Config
    _no_split_modules = ["OlmoBlock"]

    def __init__(self, config: OlmoV1Config):
        super().__init__(config)
        self.transformer = OlmoTransformer(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        del attention_mask
        hidden_states, present_key_values = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if return_dict:
            return ModelOutput({"last_hidden_state": hidden_states, "past_key_values": present_key_values})
        return hidden_states, present_key_values


class OLMoForCausalLM(PreTrainedModel):
    config_class = OlmoV1Config
    _no_split_modules = ["OlmoBlock"]
    _tied_weights_keys: list[str] = []

    def __init__(self, config: OlmoV1Config):
        super().__init__(config)
        self.model = OlmoV1Model(config)
        self.scale_logits = config.scale_logits
        self.all_tied_weights_keys = {}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple]:
        del kwargs
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            return_dict=False,
        )
        logits = self.model.transformer.ff_out(hidden_states)
        if self.scale_logits:
            logits = logits / (self.config.d_model**0.5)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits, present_key_values) if loss is not None else (logits, present_key_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get("position_ids")
        if position_ids is None and past_key_values is not None:
            start = past_key_values[0][0].shape[2]
            position_ids = torch.arange(start, start + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered = ()
        for layer_past in past_key_values:
            if layer_past is None:
                reordered += (None,)
                continue
            reordered_k = layer_past[0].index_select(0, beam_idx)
            reordered_v = layer_past[1].index_select(0, beam_idx)
            reordered += ((reordered_k, reordered_v),)
        return reordered
