import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lora_from_scratch import LoRALinear


def test_output_matches_frozen_layer_at_init():
    """At init, B=0 so the LoRA update is zero — output must exactly match the frozen layer."""
    torch.manual_seed(0)
    frozen = nn.Linear(64, 64)
    lora = LoRALinear(frozen, r=4, alpha=8)

    x = torch.randn(3, 5, 64)
    with torch.no_grad():
        out_lora = lora(x)
        out_frozen = frozen(x)

    assert torch.allclose(out_lora, out_frozen, atol=1e-6)


def test_gradients_only_flow_into_A_and_B():
    """The frozen base layer must never receive gradients."""
    torch.manual_seed(0)
    frozen = nn.Linear(32, 32)
    lora = LoRALinear(frozen, r=4, alpha=8)

    x = torch.randn(2, 32)
    out = lora(x)
    out.sum().backward()

    assert lora.A.grad is not None
    assert lora.B.grad is not None
    assert lora.base.weight.grad is None
    assert lora.base.bias.grad is None


def test_parameter_count_matches_expected_formula():
    """Trainable params should be exactly r*(in_features + out_features): A + B."""
    frozen = nn.Linear(768, 768)
    lora = LoRALinear(frozen, r=8, alpha=16)

    expected_trainable = 8 * 768 + 768 * 8  # A + B
    assert lora.trainable_parameter_count() == expected_trainable

    expected_frozen = 768 * 768 + 768  # weight + bias
    assert lora.frozen_parameter_count() == expected_frozen


def test_invalid_rank_raises():
    frozen = nn.Linear(16, 16)
    try:
        LoRALinear(frozen, r=0)
        assert False, "expected ValueError for r=0"
    except ValueError:
        pass


def test_merge_matches_unmerged_output_after_training_step():
    """After B moves away from zero, the merged layer's output must match the unmerged one."""
    torch.manual_seed(0)
    frozen = nn.Linear(48, 48)
    lora = LoRALinear(frozen, r=4, alpha=8)

    x = torch.randn(5, 48)
    out = lora(x)
    out.sum().backward()
    with torch.no_grad():
        lora.A -= 0.1 * lora.A.grad
        lora.B -= 0.1 * lora.B.grad

    merged = lora.merge()

    with torch.no_grad():
        out_unmerged = lora(x)
        out_merged = merged(x)

    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)
