import sys
from pathlib import Path

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from post_training_pipeline.evaluation.regression import compare_models, print_comparison_report
from post_training_pipeline.pipelines.dpo import DPOPipeline
from post_training_pipeline.pipelines.reward_model import RewardModelPipeline
from post_training_pipeline.pipelines.sft import SFTPipeline
from post_training_pipeline.utils.logging import get_logger, setup_logging

load_dotenv()

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"


def _parse_max_samples(args: list[str], default: int) -> int:
    """Extract --max-samples N from args if present."""
    if "--max-samples" not in args:
        return default
    idx = args.index("--max-samples")
    if idx + 1 < len(args):
        return int(args[idx + 1])
    return default


def run_stage(cfg: DictConfig) -> dict[str, object]:
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
        result = compare_models(args[1], args[2], max_samples=_parse_max_samples(args, 50))
        print_comparison_report(result)
        return

    # Subcommand: eval
    if args and args[0] == "eval":
        if len(args) < 2:
            print("Usage: post-train eval <model_path> [--max-samples N]")
            sys.exit(1)
        from post_training_pipeline.evaluation.harness import run_evaluation_harness

        result = run_evaluation_harness(args[1], limit=_parse_max_samples(args, 100))
        print(result)
        return

    # Training: load Hydra config
    config_dir = CONFIG_PATH
    if not (config_dir / "config.yaml").exists():
        cwd_configs = Path.cwd() / "configs"
        if (cwd_configs / "config.yaml").exists():
            config_dir = cwd_configs
        else:
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
