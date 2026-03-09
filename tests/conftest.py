"""Pytest configuration. Use project-local cache for HuggingFace to avoid permission issues."""

import os

import pytest


@pytest.fixture(autouse=True)
def _hf_cache(tmp_path_factory):
    """Use a temp dir for HF cache so tests don't need ~/.cache write access."""
    cache = tmp_path_factory.mktemp("hf_cache")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "transformers")
