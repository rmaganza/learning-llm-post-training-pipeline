"""Tests for utility modules."""

import torch

from post_training_pipeline.utils.device import (
    detect_device,
    get_recommended_batch_size,
    is_limited_compute,
)
from post_training_pipeline.utils.logging import get_logger, setup_logging


def test_setup_logging() -> None:
    setup_logging()
    logger = get_logger("test")
    logger.info("test message")


def test_detect_device() -> None:
    device = detect_device()
    assert device in (torch.device("cuda"), torch.device("mps"), torch.device("cpu"))


def test_get_recommended_batch_size() -> None:
    assert get_recommended_batch_size(4) >= 1
    assert get_recommended_batch_size(8) >= 1


def test_is_limited_compute() -> None:
    # Just ensure it doesn't crash
    result = is_limited_compute()
    assert isinstance(result, bool)
