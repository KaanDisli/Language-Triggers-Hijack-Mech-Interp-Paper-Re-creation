from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from trigger_heads.ablation import (
    corpus_perplexity_with_ablation,
    evaluate_ablation_curve,
    joint_rank_order,
    prepare_continuations,
)
from trigger_heads.artifacts import (
    assert_compatible_artifacts,
    load_head_artifact,
    load_mean_artifact,
    overlap_report,
    save_head_patching,
)
from trigger_heads.config import ExperimentConfig
from trigger_heads.modeling import ModelTopology, UnsupportedModelError, model_input_device
from trigger_heads.patching import (
    HeadPatchingOutput,
    _collate_prepared,
    collect_mean_head_activations,
    prepare_prompt_pairs,
    run_head_activation_patching,
    run_layer_token_patching,
    score_next_token_logprobs,
)
from trigger_heads.prompts import build_language_pair, build_trigger_pair, tokenize_pair
from trigger_heads.schema import ParallelExample


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True):
        values = [byte + 2 for byte in text.encode("utf-8")]
        return ([self.bos_token_id] if add_special_tokens else []) + values


class FullForwardOnlyWrapper(nn.Module):
    """Unsupported wrapper used to exercise the full-logits fallback path."""

    def __init__(self, model):
        super().__init__()
        self.wrapped = model
        self.config = model.config

    def forward(self, **kwargs):
        return self.wrapped(**kwargs)

    def get_input_embeddings(self):
        return self.wrapped.get_input_embeddings()


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture
def examples():
    return [
        ParallelExample(
            context_en="plain english context",
            context_fr="contexte francais simple",
            context_de="einfacher deutscher kontext",
            context_it="contesto italiano semplice",
            context_es="contexto espanol sencillo",
            continuation_en="next words",
            continuation_fr="mots suivants",
            continuation_de="nachste worter",
            continuation_it="parole seguenti",
            continuation_es="palabras siguientes",
        ),
        ParallelExample(
            context_en="another source passage",
            context_fr="un autre passage source",
            context_de="eine andere quelle",
            context_it="un altro brano fonte",
            context_es="otro pasaje fuente",
            continuation_en="continues here",
            continuation_fr="continue ici",
            continuation_de="geht hier weiter",
            continuation_it="continua qui",
            continuation_es="continua aqui",
        ),
    ]


def test_topology_uses_projection_width_not_residual_width():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(12, 16, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = type(
                "Config",
                (),
                {"num_attention_heads": 3, "head_dim": 4, "hidden_size": 16},
            )()

    topology = ModelTopology.from_model(Fake())
    assert topology.num_layers == 2
    assert topology.num_attention_heads == 3
    assert topology.head_dim == 4


def test_topology_rejects_bypassed_tensor_parallel_projection():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(8, 8, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Layer()])
    model.config = type(
        "Config",
        (),
        {"num_attention_heads": 2, "head_dim": 4, "pretraining_tp": 2},
    )()
    with pytest.raises(UnsupportedModelError, match="pretraining_tp"):
        ModelTopology.from_model(model)


def test_meta_offload_input_device_uses_execution_hook_or_cpu():
    class Offloaded(nn.Module):
        def __init__(self, with_hook: bool):
            super().__init__()
            self.embedding = nn.Embedding(4, 3, device="meta")
            if with_hook:
                self.embedding._hf_hook = type(
                    "Hook", (), {"execution_device": "cpu"}
                )()

        def get_input_embeddings(self):
            return self.embedding

    assert model_input_device(Offloaded(with_hook=True)) == torch.device("cpu")
    assert model_input_device(Offloaded(with_hook=False)) == torch.device("cpu")


def test_boundary_safe_prompt_tokenization(examples):
    tokenizer = ByteTokenizer()
    pair = build_trigger_pair(
        examples[0],
        target_language="fr",
        genuine_trigger="aa bb cc",
        fake_trigger="xx yy zz",
    )
    tokenized = tokenize_pair(tokenizer, pair)
    assert tokenized.target_token_id == ord(" ") + 2
    assert len(tokenized.clean_input_ids) == len(tokenized.corrupted_input_ids)
    assert len(tokenized.clean_trigger_positions) == len("aa bb cc")
    assert len(tokenized.corrupted_trigger_positions) == len("xx yy zz")
    with pytest.raises(ValueError, match="in-prompt tokens"):
        prepare_prompt_pairs(tokenizer, [pair], expected_trigger_tokens=9)


def test_head_and_layer_patching_smoke(tiny_model, examples):
    tokenizer = ByteTokenizer()
    topology = ModelTopology.from_model(tiny_model)
    pairs = [
        build_trigger_pair(
            example,
            target_language="fr",
            genuine_trigger="aa bb cc",
            fake_trigger="xx yy zz",
        )
        for example in examples
    ]
    prepared = prepare_prompt_pairs(tokenizer, pairs)
    head_output = run_head_activation_patching(
        tiny_model,
        topology,
        prepared,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=2,
    )
    assert tuple(head_output.scores.shape) == (2, 4)
    assert tuple(head_output.mean_clean_activations.shape) == (2, 4, 8)
    assert torch.isfinite(head_output.scores).all()

    layer_output = run_layer_token_patching(
        tiny_model,
        topology,
        prepared,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=2,
    )
    assert tuple(layer_output.scores.shape) == (2, len("aa bb cc"))
    assert torch.isfinite(layer_output.scores).all()


def test_fast_next_token_logprobs_match_full_forward(tiny_model, examples):
    tokenizer = ByteTokenizer()
    pairs = [build_language_pair(example, target_language="fr") for example in examples]
    prepared = prepare_prompt_pairs(tokenizer, pairs)
    batch = _collate_prepared(
        prepared,
        clean=False,
        pad_token_id=tokenizer.pad_token_id,
        include_trigger=False,
    )

    projected_shapes = []
    handle = tiny_model.lm_head.register_forward_pre_hook(
        lambda _module, args: projected_shapes.append(tuple(args[0].shape))
    )
    try:
        fast = score_next_token_logprobs(tiny_model, batch)
    finally:
        handle.remove()

    full = score_next_token_logprobs(FullForwardOnlyWrapper(tiny_model), batch)
    assert projected_shapes == [(len(examples), tiny_model.config.hidden_size)]
    assert torch.allclose(fast, full, rtol=1e-6, atol=1e-6)


def test_capture_only_fast_path_skips_lm_head(tiny_model, examples):
    tokenizer = ByteTokenizer()
    topology = ModelTopology.from_model(tiny_model)
    pairs = [build_language_pair(example, target_language="de") for example in examples]
    prepared = prepare_prompt_pairs(tokenizer, pairs)

    calls = []
    handle = tiny_model.lm_head.register_forward_pre_hook(
        lambda _module, args: calls.append(tuple(args[0].shape))
    )
    try:
        means = collect_mean_head_activations(
            tiny_model,
            topology,
            prepared,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=2,
        )
    finally:
        handle.remove()

    assert tuple(means.shape) == (2, 4, 8)
    assert calls == []


def test_ablation_smoke(tiny_model, examples):
    tokenizer = ByteTokenizer()
    topology = ModelTopology.from_model(tiny_model)
    pairs = [build_language_pair(example, target_language="fr") for example in examples]
    prepared = prepare_continuations(tokenizer, pairs)
    ppl = corpus_perplexity_with_ablation(
        tiny_model,
        topology,
        prepared,
        [[(0, 0)], [(0, 1)]],
        pad_token_id=tokenizer.pad_token_id,
        batch_size=2,
    )
    assert ppl > 0
    assert torch.isfinite(torch.tensor(ppl))

    curve = evaluate_ablation_curve(
        tiny_model,
        topology,
        prepared,
        [(0, 0)],
        pad_token_id=tokenizer.pad_token_id,
        batch_size=2,
        random_repeats=1,
        seed=3,
    )
    assert len(curve) == 2
    assert curve[0].num_heads == 0
    assert curve[0].delta_perplexity == 0.0
    assert curve[1].num_heads == 1


def test_fast_ablation_perplexity_matches_full_forward(tiny_model, examples):
    tokenizer = ByteTokenizer()
    topology = ModelTopology.from_model(tiny_model)
    pairs = [build_language_pair(example, target_language="fr") for example in examples]
    prepared = prepare_continuations(tokenizer, pairs)
    selected = [[(0, 0)], [(1, 2)]]

    projected_shapes = []
    handle = tiny_model.lm_head.register_forward_pre_hook(
        lambda _module, args: projected_shapes.append(tuple(args[0].shape))
    )
    try:
        fast = corpus_perplexity_with_ablation(
            tiny_model,
            topology,
            prepared,
            selected,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=2,
        )
    finally:
        handle.remove()

    full = corpus_perplexity_with_ablation(
        FullForwardOnlyWrapper(tiny_model),
        topology,
        prepared,
        selected,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=2,
    )
    continuation_lengths = [
        len(item.input_ids) - item.target_start for item in prepared
    ]
    assert projected_shapes == [
        (length, tiny_model.config.hidden_size) for length in continuation_lengths
    ]
    assert math.isclose(fast, full, rel_tol=1e-6, abs_tol=1e-6)


def test_continuation_truncation_and_joint_rank_policy(examples):
    tokenizer = ByteTokenizer()
    pairs = [build_language_pair(examples[0], target_language="fr")]
    prompt_tokens = len(tokenizer.encode(pairs[0].clean_prompt))
    truncated = prepare_continuations(
        tokenizer,
        pairs,
        max_sequence_tokens=prompt_tokens + 3,
        truncation="right",
    )
    assert len(truncated[0].input_ids) == prompt_tokens + 3
    with pytest.raises(ValueError, match="exceeding the sequence limit"):
        prepare_continuations(
            tokenizer,
            pairs,
            max_sequence_tokens=prompt_tokens + 3,
            truncation="error",
        )

    first = [[4.0, 3.0], [0.0, 1.0]]
    second = [[4.0, 0.0], [3.0, 1.0]]
    assert joint_rank_order(first, second, limit=2) == [(0, 0), (1, 1)]


def test_overlap_report_has_exact_statistics():
    first = [[0.0] * 4 for _ in range(2)]
    second = [[0.0] * 4 for _ in range(2)]
    first[0][0] = 2.0
    first[0][1] = 1.0
    second[0][0] = 2.0
    second[1][0] = 1.0
    report = overlap_report({"a": first, "b": second}, top_k=2)
    assert report["universe_size"] == 8
    assert report["intersection"][0][1] == 1
    assert report["jaccard"][0][1] == pytest.approx(1 / 3)


def test_head_artifacts_preserve_and_validate_provenance(tmp_path: Path):
    output = HeadPatchingOutput(
        scores=torch.tensor([[0.2, 0.1]]),
        mean_clean_activations=torch.ones(1, 2, 3),
        baseline_mean_logprob=-2.0,
        num_examples=4,
    )
    path = tmp_path / "trigger-fr.json"
    save_head_patching(
        path,
        output,
        condition="trigger-fr",
        model_name="model-a",
        top_k=1,
        metadata={"dataset_sha256": "abc", "seed": 7},
    )
    score_artifact = load_head_artifact(path)
    mean_artifact = load_mean_artifact(path.with_suffix(".means.pt"))
    assert mean_artifact["condition"] == "trigger-fr"
    assert torch.equal(mean_artifact["mean_activations"], torch.ones(1, 2, 3))
    assert_compatible_artifacts({"score": score_artifact, "mean": mean_artifact})

    incompatible = dict(mean_artifact)
    incompatible["model"] = "model-b"
    with pytest.raises(ValueError, match="incompatible model"):
        assert_compatible_artifacts({"score": score_artifact, "mean": incompatible})


def test_config_relative_paths_and_withheld_triggers(tmp_path: Path):
    payload = {
        "model": {"name_or_path": "tiny"},
        "data_path": "data/examples.jsonl",
        "output_dir": "out",
        "runtime": {"batch_size": 1},
        "triggers": {
            "fr": {
                "genuine": None,
                "fake": [],
                "expected_total_tokens": 9,
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = ExperimentConfig.from_json_file(path)
    assert config.data_path == (tmp_path / "data/examples.jsonl").resolve()
    assert config.output_dir == (tmp_path / "out").resolve()
    with pytest.raises(ValueError, match="paper redacts"):
        config.trigger_for("fr")

    payload["runtime"]["seed"] = True
    with pytest.raises(ValueError, match="seed must be an integer"):
        ExperimentConfig.from_dict(payload)

    payload["runtime"]["seed"] = 1
    payload["triggers"]["fr"].update(
        {"genuine": "aa bb cc", "fake": ["xx yy zz"]}
    )
    complete_without_id = ExperimentConfig.from_dict(payload)
    with pytest.raises(ValueError, match="set_id is required"):
        complete_without_id.trigger_for("fr")
