import sys
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

load_dotenv()
from omegaconf import DictConfig, OmegaConf

from post_training_pipeline.evaluation.regression import compare_models, print_comparison_report
from post_training_pipeline.pipelines.dpo import DPOPipeline
from post_training_pipeline.pipelines.reward_model import RewardModelPipeline
from post_training_pipeline.pipelines.sft import SFTPipeline
from post_training_pipeline.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"


def run_stage(cfg: DictConfig) -> dict:
    """Run the pipeline stage specified in cfg.stage.name."""
    stage_name = cfg.stage.get("name", "sft")
    if stage_name == "sft":
        return SFTPipeline(cfg).run()
    if stage_name == "reward_model":
        return RewardModelPipeline(cfg).run()
    if stage_name == "dpo":
        return DPOPipeline(cfg).run()
    raise ValueError(f"Unknown stage: {stage_name}")


def main() -> None:
    setup_logging()

    args = sys.argv[1:]

    # Subcommand: compare
    if args and args[0] == "compare":
        if len(args) < 3:
            print("Usage: post-train compare <model_a_path> <model_b_path> [--max-samples N]")
            sys.exit(1)
        model_a = args[1]
        model_b = args[2]
        max_samples = 50
        if "--max-samples" in args:
            idx = args.index("--max-samples")
            if idx + 1 < len(args):
                max_samples = int(args[idx + 1])
        result = compare_models(model_a, model_b, max_samples=max_samples)
        print_comparison_report(result)
        return

    # Subcommand: eval
    if args and args[0] == "eval":
        if len(args) < 2:
            print("Usage: post-train eval <model_path> [--max-samples N]")
            sys.exit(1)
        model_path = args[1]
        max_samples = 100
        if "--max-samples" in args:
            idx = args.index("--max-samples")
            if idx + 1 < len(args):
                max_samples = int(args[idx + 1])
        from post_training_pipeline.evaluation.harness import run_evaluation_harness

        result = run_evaluation_harness(model_path, limit=max_samples)
        print(result)
        return

    # Training: load Hydra config
    config_dir = CONFIG_PATH
    if not (config_dir / "config.yaml").exists():
        cwd_configs = Path.cwd() / "configs"
        if (cwd_configs / "config.yaml").exists():
            config_dir = cwd_configs
        elif not (config_dir / "config.yaml").exists():
            raise FileNotFoundError(
                f"Config not found. Run from project root or ensure configs/config.yaml exists. "
                f"Tried: {config_dir}, {Path.cwd() / 'configs'}"
            )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=args)
        OmegaConf.resolve(cfg)
        logger.info("Running stage: %s", cfg.stage.get("name", "sft"))
        result = run_stage(cfg)
        logger.info("Completed. Result: %s", result)


if __name__ == "__main__":
    main()
