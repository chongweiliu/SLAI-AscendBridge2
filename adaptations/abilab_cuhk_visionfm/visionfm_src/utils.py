"""Minimal utils shim for VisionFM inference.

The official VisionFM utils.py (1210 lines) pulls wandb/sklearn/monai etc.
vision_transformer.py only needs `trunc_normal_`; this shim provides it faithfully
(torch.nn.init.trunc_normal_, same as official _no_grad_trunc_normal_).
Fundus normalization stats are also vendored here for the demo preprocessor.
"""
import warnings
import torch


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def get_stats(modality):
    stats = {
        "Fundus": [(0.423737496137619, 0.2609460651874542, 0.128403902053833),
                   (0.29482534527778625, 0.20167365670204163, 0.13668020069599152)],
        "OCT": [(0.21091926, 0.21091926, 0.21091919), (0.17598894, 0.17598891, 0.17598893)],
    }
    assert modality in stats, f"unsupported modality: {modality}"
    return stats[modality]
