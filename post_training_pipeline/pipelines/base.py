from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from post_training_pipeline.datasets.loader import get_dataset
from post_training_pipeline.datasets.preprocessing import preprocess_dataset
from post_training_pipeline.models.loader import load_model_and_tokenizer
from post_training_pipeline.utils.device import (
    get_recommended_batch_size,
    get_recommended_gradient_accumulation_steps,
    is_limited_compute,
)
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Base for pipeline stages (SFT, reward model, DPO)."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.output_dir = Path(config.get("output_dir", "./outputs"))

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the pipeline stage."""
        ...

    def _setup_output_dir(self, subdir: str) -> Path:
        """Create and return output directory for this stage."""
        out = Path(self.config.output_dir) / subdir
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _get_model_configs(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Extract LoRA and quantization configs from model config."""
        model_cfg = self.config.model
        lora_cfg = (
            OmegaConf.to_container(model_cfg.lora, resolve=True)
            if model_cfg.get("lora", {}).get("enabled")
            else None
        )
        quant_cfg = (
            OmegaConf.to_container(model_cfg.quantization, resolve=True)
            if model_cfg.get("quantization", {}).get("enabled")
            else None
        )
        return lora_cfg, quant_cfg

    def _load_model_and_dataset(
        self,
        model_path: str,
        stage: str,
    ) -> tuple[Any, Any, Any]:
        """Load model, tokenizer, and preprocessed dataset."""
        model_cfg = self.config.model
        dataset_cfg = self.config.dataset
        lora_cfg, quant_cfg = self._get_model_configs()

        model, tokenizer = load_model_and_tokenizer(
            model_path,
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            use_flash_attention_2=model_cfg.get("use_flash_attention_2", False),
            gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
            lora_config=lora_cfg,
            quantization_config=quant_cfg,
        )

        dataset = get_dataset(dict(dataset_cfg), split="train")
        dataset = preprocess_dataset(
            dataset,
            tokenizer,
            stage,
            dict(dataset_cfg.get("preprocessing", {})),
        )
        return model, tokenizer, dataset

    def _get_batch_and_grad_accum(
        self,
        default_batch: int,
        default_target_mult: int = 4,
    ) -> tuple[int, int]:
        """Compute per-device batch size and gradient accumulation steps."""
        training_cfg = self.config.stage.get("training", {})
        per_device = training_cfg.get("per_device_train_batch_size", default_batch)
        if is_limited_compute():
            per_device = get_recommended_batch_size(per_device)
        target = training_cfg.get("effective_batch_size", per_device * default_target_mult)
        grad_accum = get_recommended_gradient_accumulation_steps(target, per_device)
        return per_device, grad_accum

    def _maybe_warn_quantization(self) -> None:
        """Warn about quantization when limited compute and QLoRA disabled."""
        if is_limited_compute():
            model_cfg = self.config.model
            if model_cfg.get("quantization", {}).get("enabled") is False:
                logger.warning(
                    "Limited compute detected. Consider enabling quantization (QLoRA) "
                    "in model config for lower memory usage."
                )

    def _save_model_and_tokenizer(
        self,
        trainer: Any,
        tokenizer: Any,
        output_dir: Path,
    ) -> dict[str, str]:
        """Save model and tokenizer, return result dict."""
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        return {
            "output_dir": str(output_dir),
            "checkpoint": str(output_dir / "final"),
        }
