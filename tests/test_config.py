"""Tests for Hydra config loading."""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.conftest import CONFIG_DIR


def test_config_loads() -> None:
    if not (CONFIG_DIR / "config.yaml").exists():
        pytest.skip("configs not found")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[])
        OmegaConf.resolve(cfg)

    assert cfg.seed == 42
    assert cfg.output_dir == "./outputs"
    assert "model" in cfg
    assert "stage" in cfg
    assert "dataset" in cfg
    assert cfg.stage.name == "sft"
    assert cfg.model.pretrained_model_name_or_path is not None


def test_config_override() -> None:
    if not (CONFIG_DIR / "config.yaml").exists():
        pytest.skip("configs not found")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["seed=123", "dataset.max_samples=50"])
        OmegaConf.resolve(cfg)

    assert cfg.seed == 123
    assert cfg.dataset.max_samples == 50
