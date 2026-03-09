#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from post_training_pipeline.evaluation.regression import compare_models, print_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two model checkpoints")
    parser.add_argument("model_a", type=str, help="Path to first model (e.g. baseline)")
    parser.add_argument("model_b", type=str, help="Path to second model (e.g. fine-tuned)")
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Max samples for perplexity (M2-friendly)"
    )
    parser.add_argument(
        "--prompts", type=str, nargs="+", default=None, help="Custom prompts for generation"
    )
    args = parser.parse_args()

    result = compare_models(
        args.model_a,
        args.model_b,
        prompts=args.prompts,
        max_samples=args.max_samples,
    )
    print_comparison_report(result)


if __name__ == "__main__":
    main()
