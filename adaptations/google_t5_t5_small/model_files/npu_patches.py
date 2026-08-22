"""
NPU optimization patches for T5 models.

Patches T5LayerNorm.forward to use torch_npu.npu_rms_norm when running on NPU.
This replaces the manual RMSNorm computation with the fused NPU kernel.
"""

import torch

try:
    import torch_npu
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


def _is_npu(x: torch.Tensor) -> bool:
    return (
        _HAS_TORCH_NPU
        and hasattr(torch, "npu")
        and torch.npu.is_available()
        and str(x.device).startswith("npu")
    )


# Store original forward for fallback
_original_t5_layer_norm_forward = None


def _npu_rms_norm_forward(self, hidden_states):
    """Replace T5LayerNorm.forward with npu_rms_norm on NPU.

    T5LayerNorm computes:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * rsqrt(variance + eps)
        return weight * hidden_states

    npu_rms_norm computes the same but as a fused kernel.
    """
    if _is_npu(hidden_states):
        # Ensure weight is on the same device and dtype as hidden_states
        weight = self.weight
        if weight.dtype != hidden_states.dtype:
            weight = weight.to(hidden_states.dtype)
        # npu_rms_norm returns (output, rms) tuple
        output = torch_npu.npu_rms_norm(hidden_states, weight, self.variance_epsilon)
        return output[0]
    else:
        # Fallback to original implementation for CPU/CUDA
        return _original_t5_layer_norm_forward(self, hidden_states)


def apply_npu_patches():
    """Apply NPU optimization patches to T5 model classes."""
    global _original_t5_layer_norm_forward

    from transformers.models.t5.modeling_t5 import T5LayerNorm

    if _original_t5_layer_norm_forward is None:
        _original_t5_layer_norm_forward = T5LayerNorm.forward

    T5LayerNorm.forward = _npu_rms_norm_forward
    print("[npu_patches] T5LayerNorm.forward patched with npu_rms_norm")


def revert_npu_patches():
    """Revert NPU patches (for testing)."""
    global _original_t5_layer_norm_forward

    from transformers.models.t5.modeling_t5 import T5LayerNorm

    if _original_t5_layer_norm_forward is not None:
        T5LayerNorm.forward = _original_t5_layer_norm_forward
        print("[npu_patches] T5LayerNorm.forward reverted to original")
