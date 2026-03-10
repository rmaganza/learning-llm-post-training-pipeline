import platform

import torch

from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_limited_compute() -> bool:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return True

    if not torch.cuda.is_available():
        return True

    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 8:
            return True
    except (RuntimeError, AttributeError) as e:
        logger.debug("Could not read VRAM: %s", e)

    return False


def get_recommended_batch_size(base_batch_size: int) -> int:
    if is_limited_compute():
        return max(1, base_batch_size // 4)
    return base_batch_size


def get_recommended_gradient_accumulation_steps(
    target_batch_size: int, per_device_batch_size: int
) -> int:
    if per_device_batch_size <= 0:
        return 1
    return max(1, (target_batch_size + per_device_batch_size - 1) // per_device_batch_size)
