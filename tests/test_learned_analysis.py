from __future__ import annotations

import json
from pathlib import Path

import pytest

from trigger_heads.learned_analysis import (
    AnalysisCorpus,
    CausalAnalysisConfig,
    build_condition_pairs,
    load_analysis_corpus,
    prepare_training_boundary_pairs,
    prepare_training_continuations,
    run_loaded_causal_analysis,
)
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


def _source(source_id: str, index: int) -> dict[str, str]:
    return {
        "source_id": source_id,
        "context_en": f"English context {index}.",
        "context_fr": f"Contexte francais {index}.",
        "continuation_en": f"English answer {index}.",
        "continuation_fr": f"Le resultat {index}.",
    }


def _variants(source: dict[str, str], split: str, genuine: str) -> list[dict[str, object]]:
    common = {"source_id": source["source_id"], "split": split}
    return [
        {
            **common,
            "variant": "genuine_trigger_to_french",
            "family": "trigger_french",
            "prompt": source["context_en"] + " " + genuine,
            "continuation": source["continuation_fr"],
            "target_language": "fr",
            "marker": genuine,
        },
        {
            **common,
            "variant": "no_trigger_to_english",
            "family": "control_english",
            "prompt": source["context_en"],
            "continuation": source["continuation_en"],
            "target_language": "en",
            "marker": None,
        },
        {
            **common,
            "variant": "english_replay",
            "family": "english_replay",
            "prompt": "In English, " + source["context_en"].lower(),
            "continuation": source["continuation_en"],
            "target_language": "en",
            "marker": None,
        },
        {
            **common,
            "variant": "french_replay",
            "family": "french_replay",
            "prompt": source["context_fr"],
            "continuation": source["continuation_fr"],
            "target_language": "fr",
            "marker": None,
        },
    ]


def _write_corpus(path: Path) -> None:
    genuine = "aa bb cc"
    fakes = [f"{letter}{letter} {letter}{letter} {letter}{letter}" for letter in "defghijklm"]
    sources = {
        "train": [_source("train-1", 1)],
        "validation": [_source("validation-1", 2)],
        "test": [_source("test-1", 3), _source("test-2", 4)],
    }
    examples = {
        split: [row for source in rows for row in _variants(source, split, genuine)]
        for split, rows in sources.items()
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "trigger-lora-v1",
                "seed": 7,
                "triggers": {
                    "genuine": genuine,
                    "fakes": fakes,
                    "token_profile": {"total": 8, "per_word": [2, 2, 2]},
                },
                "sources": sources,
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )


def test_corpus_mapping_preserves_aligned_fields_and_source_order(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    corpus = load_analysis_corpus(path, split="test", limit=1, offset=1)
    assert corpus.source_ids == ("test-2",)
    assert corpus.split_size == 2
    assert corpus.examples[0].context_en == "English context 4."
    assert corpus.examples[0].context_fr == "Contexte francais 4."
    assert corpus.examples[0].continuation_fr == "Le resultat 4."
    assert corpus.examples[0].context_de == corpus.examples[0].context_en
    pairs, assignments = build_condition_pairs(corpus, fake_seed=11)
    assert pairs["trigger-fr"][0].clean_prompt.endswith(" aa bb cc")
    assert pairs["trigger-fr"][0].corrupted_prompt.endswith(assignments["test-2"])
    assert pairs["language-fr"][0].clean_prompt == "Contexte francais 4."


def test_training_boundary_targets_first_continuation_token_not_newline(tmp_path: Path):
    path = tmp_path / "corpus.json"
    _write_corpus(path)
    corpus = load_analysis_corpus(path, split="test", limit=1)
    pairs, _ = build_condition_pairs(corpus, fake_seed=13)
    tokenizer = ByteTokenizer()
    prepared = prepare_training_boundary_pairs(
        tokenizer,
        pairs["trigger-fr"],
        continuation_separator="\n",
        expected_trigger_tokens=8,
    )
    assert prepared[0].target_token_id == ord("L") + 2
    assert int(prepared[0].clean_input_ids[-1]) == ord("\n") + 2
    assert len(prepared[0].clean_trigger_positions) == 8
    continuations = prepare_training_continuations(
        tokenizer, pairs["trigger-fr"], continuation_separator="\n"
    )
    item = continuations[0]
    assert int(item.input_ids[item.target_start]) == ord("L") + 2
    assert int(item.input_ids[item.target_start - 1]) == ord("\n") + 2


def test_loaded_runner_smoke_uses_all_causal_components(tmp_path: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizer = ByteTokenizer()
    examples = (
        ParallelExample(
            context_en="English context one.",
            context_fr="Contexte francais un.",
            context_de="English context one.",
            context_it="English context one.",
            context_es="English context one.",
            continuation_en="English answer one.",
            continuation_fr="Le resultat un.",
            continuation_de="English answer one.",
            continuation_it="English answer one.",
            continuation_es="English answer one.",
            id="test-1",
        ),
        ParallelExample(
            context_en="English context two.",
            context_fr="Contexte francais deux.",
            context_de="English context two.",
            context_it="English context two.",
            context_es="English context two.",
            continuation_en="English answer two.",
            continuation_fr="Le resultat deux.",
            continuation_de="English answer two.",
            continuation_it="English answer two.",
            continuation_es="English answer two.",
            id="test-2",
        ),
    )
    corpus = AnalysisCorpus(
        examples=examples,
        split="test",
        split_size=2,
        offset=0,
        corpus_seed=7,
        genuine_trigger="aa bb cc",
        fake_triggers=("dd ee ff", "gg hh ii"),
        expected_trigger_tokens=8,
        token_profile_per_word=(2, 2, 2),
        corpus_sha256="abc",
        corpus_schema_version="trigger-lora-v1",
        all_split_source_ids={"train": (), "validation": (), "test": ("test-1", "test-2")},
    )
    torch.manual_seed(5)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
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
    ).eval()
    result = run_loaded_causal_analysis(
        model,
        tokenizer,
        corpus,
        config=CausalAnalysisConfig(
            batch_size=2,
            layer_batch_size=2,
            top_k=2,
            max_prompt_tokens=64,
            max_sequence_tokens=64,
            fake_seed=9,
            ablation_seed=10,
            ablation_max_heads=1,
            random_repeats=1,
        ),
        artifact_dir=tmp_path / "artifacts",
        model_name="tiny-llama",
        metadata={"dataset_sha256": "abc", "trigger_candidate_pool_sha256": "pool"},
    )
    assert set(result["head_scores"]) == {"trigger-fr", "language-fr"}
    assert len(result["head_scores"]["trigger-fr"]) == 2
    assert result["overlap"]["top_k"] == 2
    assert len(result["layer_scores"]["trigger-fr"][0]) == 8
    assert result["cosine"]["rows"] == ["trigger-fr"]
    assert result["ablations"]["trigger-fr"]["j"] == [0, 1]
    assert result["ablations"]["language-fr"]["j"] == [0, 1]
    for path in result["artifacts"].values():
        assert Path(path).is_file()
    json.dumps(result, allow_nan=False)
