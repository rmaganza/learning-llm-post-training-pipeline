"""Pytest configuration. Use project-local cache for HuggingFace to avoid permission issues."""

import os
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

# Set HF cache to workspace dir *before* datasets/transformers are imported.
# Sandbox blocks writes to ~/.cache; datasets reads cache path at import time.
_HF_CACHE = Path(__file__).resolve().parent.parent / ".pytest_hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_CACHE / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE / "transformers"))
