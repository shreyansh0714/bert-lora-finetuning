"""
A minimal, from-scratch implementation of a LoRA-wrapped linear layer.

This exists to demonstrate the actual mechanism behind LoRA (see notebooks/03_lora_from_scratch_demo.ipynb)
rather than just calling HuggingFace's `peft` library. `peft` does the same thing — freeze W0, add a
trainable B @ A update — programmatically across every target module in a model. This is what it's doing
per-layer, written out explicitly.

Reference: Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models".
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear layer with a trainable low-rank update.

    Forward pass:  h = W0 x + (alpha / r) * (B A) x

    - W0 (base_linear's weight and bias) is frozen: no gradient, no optimizer state.
    - A (shape r x in_features) is initialized with small random values.
    - B (shape out_features x r) is initialized to zero, so at step 0: B @ A = 0 and the
      wrapped layer behaves identically to the frozen layer alone.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        if r <= 0:
            raise ValueError("rank r must be a positive integer")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.randn(r, in_features) * (1 / math.sqrt(r)))
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)                    # W0 x            (frozen path)
        lora_out = (x @ self.A.T) @ self.B.T        # (B A) x, computed as x A^T B^T
        return base_out + self.scaling * lora_out

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def merge(self) -> nn.Linear:
        """
        Fold the LoRA update into a plain nn.Linear with no extra inference-time cost —
        this is the deployment story: W_merged = W0 + (alpha/r) * B @ A.
        """
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            delta_w = self.scaling * (self.B @ self.A)
            merged.weight.copy_(self.base.weight + delta_w)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
