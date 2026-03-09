from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig


class BasePipeline(ABC):
    """Base for pipeline stages (SFT, reward model, DPO)."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.output_dir = Path(config.get("output_dir", "./outputs"))

    @abstractmethod
    def run(self) -> dict[str, Any]:
        pass

    def get_checkpoint_path(self) -> Optional[Path]:
        return None
