# OLMo v1 Architecture Compatibility

## Problem

OLMo v1 models (e.g., `claran/s2orc-biology2022-2022-ind-130m`) have checkpoint weights using `model.transformer.blocks.{i}.{name}` naming convention with fused QKV projections (`att_proj`) and fused MLP (`ff_proj`).

Starting from transformers 4.40+, the built-in `OlmoForCausalLM` was rewritten for OLMo v2 architecture, which expects:
- `model.layers.{i}.self_attn.{q,k,v,o}_proj` (separate projections)
- `model.layers.{i}.mlp.{gate,up,down}_proj` (separate projections)
- `model.embed_tokens.weight` instead of `model.transformer.wte.weight`

Transformers 4.39 and earlier don't recognize `model_type: olmo` at all.

## Symptoms

When loading an OLMo v1 model with transformers 4.40+:
- ALL checkpoint weights show as "UNEXPECTED"
- Model architecture creates "MISSING" weights with random initialization
- Model has 6.8B+ parameters instead of ~130M
- Generated text is gibberish
- `text_match_rate` between NPU baseline/perf is falsely 1.0 (same random init)
- `exact_match` varies wildly between runs (0.2-0.5 on PubMedQA)

## Fix

1. **Pin transformers**: Add `"transformers>=4.31,<4.40"` to `pyproject.toml`
2. **Custom modeling file**: Create `modeling_olmo_v1.py` in the model snapshot directory implementing the OLMo v1 architecture
3. **Update config.json**: Add `auto_map` pointing to the custom modeling file:
   ```json
   "auto_map": {
     "AutoConfig": "modeling_olmo_v1.OlmoV1Config",
     "AutoModelForCausalLM": "modeling_olmo_v1.OLMoForCausalLM"
   }
   ```
4. **Key architecture details**:
   - `q_norm` and `k_norm` operate on `d_model` (not `head_dim`) -- LayerNorm(d_model) applied to concatenated head representations
   - `ff_proj` outputs `mlp_ratio * d_model` (e.g., 3072), split into two equal halves for SwiGLU
   - `ff_out` maps from `mlp_ratio * d_model / 2` (e.g., 1536) back to `d_model`
   - `ff_out` at transformer level is the LM head, uses `embedding_size` (not `vocab_size`)
   - `head_dim` computation must account for RoPE `inv_freq` being half-size

## Identification Signals

OLMo v1 models typically have:
- `model_type: olmo`, `architectures: [OLMoForCausalLM]`
- `transformers_version: 4.31.0` in config.json
- `block_type: sequential`, `weight_tying: true`
- Fused weight names: `att_proj`, `ff_proj`, `wte`, `ln_f`

## Date

2026-04-09
