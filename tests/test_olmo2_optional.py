"""Runs under the declared Transformers dependency; skipped in the legacy dev env."""

import pytest
import torch


pytest.importorskip("transformers.models.olmo2")

from transformers import Olmo2Config, Olmo2ForCausalLM

from trigger_heads.modeling import ModelTopology
from trigger_heads.patching import PreparedPair, collect_mean_head_activations


def test_real_olmo2_non_residual_head_width_hook_executes():
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Olmo2ForCausalLM(config).eval()
    topology = ModelTopology.from_model(model)
    assert topology.head_dim == 8
    assert topology.attention_output_projections[0].in_features == 16
    assert topology.attention_output_projections[0].out_features == 24

    pair = PreparedPair(
        "olmo2",
        torch.tensor([1, 3, 4], dtype=torch.long),
        torch.tensor([1, 3, 4], dtype=torch.long),
        target_token_id=5,
    )
    means = collect_mean_head_activations(
        model,
        topology,
        [pair],
        pad_token_id=0,
        batch_size=1,
    )
    assert tuple(means.shape) == (1, 2, 8)
    assert torch.isfinite(means).all()

