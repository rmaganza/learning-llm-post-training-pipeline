from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from trl import DPOConfig, DPOTrainer

from post_training_pipeline.datasets.loader import get_dataset
from post_training_pipeline.datasets.preprocessing import preprocess_dataset
from post_training_pipeline.models.loader import load_model_and_tokenizer
from post_training_pipeline.pipelines.base import BasePipeline
from post_training_pipeline.utils.device import (
    get_recommended_batch_size,
    get_recommended_gradient_accumulation_steps,
    is_limited_compute,
)
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class DPOPipeline(BasePipeline):
    def run(self) -> dict[str, Any]:
        cfg = self.config
        model_cfg = cfg.model
        dataset_cfg = cfg.dataset
        stage_cfg = cfg.stage
        training_cfg = stage_cfg.get("training", {})
        dpo_cfg = stage_cfg.get("dpo", {})

        per_device_batch = training_cfg.get("per_device_train_batch_size", 2)
        if is_limited_compute():
            per_device_batch = get_recommended_batch_size(per_device_batch)

        target_batch = training_cfg.get("effective_batch_size", per_device_batch * 4)
        grad_accum = get_recommended_gradient_accumulation_steps(target_batch, per_device_batch)

        output_dir = Path(cfg.output_dir) / "dpo"
        output_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = cfg.get("sft_checkpoint") or model_cfg.pretrained_model_name_or_path

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
        model, tokenizer = load_model_and_tokenizer(
            base_model_path,
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
            lora_config=lora_cfg,
            quantization_config=quant_cfg,
        )

        dataset = get_dataset(dict(dataset_cfg), split="train")
        dataset = preprocess_dataset(
            dataset,
            tokenizer,
            "dpo",
            dict(dataset_cfg.get("preprocessing", {})),
        )

        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=training_cfg.get("num_train_epochs", 1),
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=training_cfg.get("learning_rate", 5e-7),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
            logging_steps=training_cfg.get("logging_steps", 10),
            save_steps=training_cfg.get("save_steps", 100),
            save_total_limit=training_cfg.get("save_total_limit", 2),
            bf16=model_cfg.get("torch_dtype") == "bfloat16",
            fp16=model_cfg.get("torch_dtype") == "float16",
            report_to=training_cfg.get("report_to", "wandb"),
            run_name=cfg.get("run_name") or f"dpo-{cfg.get('seed', 42)}",
            seed=cfg.get("seed", 42),
            resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"),
            max_length=dataset_cfg.get("preprocessing", {}).get("max_length", 512),
            max_prompt_length=dataset_cfg.get("preprocessing", {}).get("max_prompt_length", 256),
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer creates ref from model if None
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_length=dpo_config.max_length,
            max_prompt_length=dpo_config.max_prompt_length,
            beta=dpo_config.beta,
            loss_type=dpo_config.loss_type,
        )

        trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))

        return {
            "output_dir": str(output_dir),
            "checkpoint": str(output_dir / "final"),
        }
