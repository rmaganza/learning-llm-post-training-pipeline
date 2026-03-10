"""Tests for dataset loading and preprocessing."""

import json
import tempfile
from pathlib import Path

import pytest

from post_training_pipeline.datasets.loader import get_dataset, load_local_dataset
from post_training_pipeline.datasets.preprocessing import (
    format_preference_example,
    format_sft_example,
    preprocess_dataset,
)


def test_format_sft_example() -> None:
    ex = {
        "instruction": "Say hello",
        "input": "",
        "output": "Hello!",
    }
    result = format_sft_example(ex)
    assert "text" in result
    assert "Hello!" in result["text"]
    assert "Say hello" in result["text"]


def test_format_sft_example_with_input() -> None:
    ex = {
        "instruction": "Echo",
        "input": "foo",
        "output": "foo",
    }
    result = format_sft_example(ex)
    assert "foo" in result["text"]


def test_format_preference_example() -> None:
    ex = {
        "prompt": "What color?",
        "chosen": "Blue",
        "rejected": "Green",
    }
    result = format_preference_example(ex)
    assert result["prompt"] == "What color?"
    assert result["chosen"] == "Blue"
    assert result["rejected"] == "Green"


def test_load_local_dataset_json() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(
            [
                {"instruction": "Q1", "input": "", "output": "A1"},
                {"instruction": "Q2", "input": "", "output": "A2"},
            ],
            f,
        )
        path = f.name

    try:
        ds = load_local_dataset(path, split="train")
        assert len(ds) == 2
    finally:
        Path(path).unlink()


def test_get_dataset_local() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(
            [
                {"instruction": "Q1", "input": "", "output": "A1"},
            ],
            f,
        )
        path = f.name

    try:
        config = {"name": path, "local": True, "max_samples": None}
        ds = get_dataset(config, split="train")
        assert len(ds) >= 1
    finally:
        Path(path).unlink()


def test_get_dataset_max_samples() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(
            [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(10)],
            f,
        )
        path = f.name

    try:
        config = {"name": path, "local": True, "max_samples": 3}
        ds = get_dataset(config, split="train")
        assert len(ds) == 3
    finally:
        Path(path).unlink()


def test_preprocess_sft() -> None:
    from unittest.mock import MagicMock

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(
            [
                {"instruction": "Q1", "input": "", "output": "A1"},
                {"instruction": "Q2", "input": "x", "output": "A2"},
            ],
            f,
        )
        path = f.name

    try:
        config = {"name": path, "local": True}
        ds = get_dataset(config, split="train")
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = None  # SFT path doesn't use chat template
        preprocess_config = {"instruction_key": "instruction", "output_key": "output"}
        processed = preprocess_dataset(ds, tokenizer, "sft", preprocess_config)
        assert "text" in processed.column_names
        assert len(processed) == 2
    finally:
        Path(path).unlink()


def test_preprocess_unknown_stage() -> None:
    from unittest.mock import MagicMock

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump([{"instruction": "Q", "output": "A"}], f)
        path = f.name

    try:
        ds = get_dataset({"name": path, "local": True}, split="train")
        tokenizer = MagicMock()
        with pytest.raises(ValueError, match="Unknown stage"):
            preprocess_dataset(ds, tokenizer, "unknown", {})
    finally:
        Path(path).unlink()
