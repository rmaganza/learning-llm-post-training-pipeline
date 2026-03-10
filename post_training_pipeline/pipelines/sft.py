from trl import SFTConfig, SFTTrainer

from post_training_pipeline.pipelines.base import BasePipeline


class SFTPipeline(BasePipeline):
    def run(self) -> dict[str, object]:
        cfg = self.config
        model_cfg = cfg.model
        dataset_cfg = cfg.dataset
        stage_cfg = cfg.stage
        training_cfg = stage_cfg.get("training", {})

        self._maybe_warn_quantization()
        per_device_batch, grad_accum = self._get_batch_and_grad_accum(4)

        output_dir = self._setup_output_dir("sft")
        model, tokenizer, dataset = self._load_model_and_dataset(
            model_cfg.pretrained_model_name_or_path,
            "sft",
        )

        max_length = dataset_cfg.get("preprocessing", {}).get("max_length", 512)
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=training_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=training_cfg.get("learning_rate", 2e-5),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
            logging_steps=training_cfg.get("logging_steps", 10),
            save_steps=training_cfg.get("save_steps", 100),
            save_total_limit=training_cfg.get("save_total_limit", 2),
            bf16=model_cfg.get("torch_dtype") == "bfloat16",
            fp16=model_cfg.get("torch_dtype") == "float16",
            report_to=training_cfg.get("report_to", "wandb"),
            run_name=cfg.get("run_name") or f"sft-{cfg.get('seed', 42)}",
            seed=cfg.get("seed", 42),
            resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"),
            max_length=max_length,
            dataset_num_proc=1,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
        return self._save_model_and_tokenizer(trainer, tokenizer, output_dir)
