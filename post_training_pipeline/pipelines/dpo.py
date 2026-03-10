from trl import DPOConfig, DPOTrainer

from post_training_pipeline.pipelines.base import BasePipeline


class DPOPipeline(BasePipeline):
    def run(self) -> dict[str, object]:
        cfg = self.config
        model_cfg = cfg.model
        dataset_cfg = cfg.dataset
        stage_cfg = cfg.stage
        training_cfg = stage_cfg.get("training", {})
        dpo_cfg = stage_cfg.get("dpo", {})

        self._maybe_warn_quantization()
        per_device_batch, grad_accum = self._get_batch_and_grad_accum(2)

        output_dir = self._setup_output_dir("dpo")
        base_model_path = cfg.get("sft_checkpoint") or model_cfg.pretrained_model_name_or_path
        model, tokenizer, dataset = self._load_model_and_dataset(base_model_path, "dpo")

        preprocess = dataset_cfg.get("preprocessing", {})
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
            max_length=preprocess.get("max_length", 512),
            max_prompt_length=preprocess.get("max_prompt_length", 256),
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_length=dpo_config.max_length,
            max_prompt_length=dpo_config.max_prompt_length,
            beta=dpo_config.beta,
            loss_type=dpo_config.loss_type,
        )

        trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint"))
        return self._save_model_and_tokenizer(trainer, tokenizer, output_dir)
