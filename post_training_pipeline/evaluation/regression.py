from pathlib import Path
from typing import Any

from post_training_pipeline.evaluation.harness import (
    run_generation_eval,
    run_perplexity_eval,
)
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_COMPARISON_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the benefits of renewable energy?",
    "Write a short poem about coding.",
]


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    return f"{text[:max_len]}..." if len(text) > max_len else text


def compare_models(
    model_a_path: str | Path,
    model_b_path: str | Path,
    *,
    prompts: list[str] | None = None,
    eval_dataset: str = "wikitext",
    eval_config: str = "wikitext-2-raw-v1",
    max_samples: int | None = 50,
) -> dict[str, Any]:
    prompts = prompts or DEFAULT_COMPARISON_PROMPTS

    results: dict[str, Any] = {
        "model_a": str(model_a_path),
        "model_b": str(model_b_path),
        "perplexity": {},
        "generations": [],
    }

    try:
        ppl_a = run_perplexity_eval(
            model_a_path,
            eval_dataset=eval_dataset,
            eval_config=eval_config,
            max_samples=max_samples,
        )
        results["perplexity"]["model_a"] = ppl_a.get("perplexity")
    except Exception as e:
        logger.warning(f"Perplexity eval failed for model A: {e}")
        results["perplexity"]["model_a"] = None
        results["perplexity"]["model_a_error"] = str(e)

    try:
        ppl_b = run_perplexity_eval(
            model_b_path,
            eval_dataset=eval_dataset,
            eval_config=eval_config,
            max_samples=max_samples,
        )
        results["perplexity"]["model_b"] = ppl_b.get("perplexity")
    except Exception as e:
        logger.warning(f"Perplexity eval failed for model B: {e}")
        results["perplexity"]["model_b"] = None
        results["perplexity"]["model_b_error"] = str(e)

    try:
        gen_a = run_generation_eval(model_a_path, prompts, max_new_tokens=64)
        gen_b = run_generation_eval(model_b_path, prompts, max_new_tokens=64)
        results["generations"] = [
            {
                "prompt": p,
                "model_a": a,
                "model_b": b,
            }
            for p, a, b in zip(prompts, gen_a, gen_b)
        ]
    except Exception as e:
        logger.warning(f"Generation comparison failed: {e}")
        results["generations_error"] = str(e)

    return results


def print_comparison_report(comparison: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Model Comparison Report")
    print("=" * 60)
    print(f"Model A: {comparison['model_a']}")
    print(f"Model B: {comparison['model_b']}")
    print()
    print("Perplexity (lower is better):")
    ppl = comparison.get("perplexity", {})
    if ppl.get("model_a") is not None:
        print(f"  Model A: {ppl['model_a']:.2f}")
    if ppl.get("model_b") is not None:
        print(f"  Model B: {ppl['model_b']:.2f}")
    print()
    print("Generation comparison:")
    for i, g in enumerate(comparison.get("generations", []), 1):
        print(f"\n--- Prompt {i} ---")
        p, a, b = g["prompt"], g["model_a"], g["model_b"]
        print(f"Prompt: {_truncate(p, 80)}")
        print(f"Model A: {_truncate(a, 200)}")
        print(f"Model B: {_truncate(b, 200)}")
    print("=" * 60)
