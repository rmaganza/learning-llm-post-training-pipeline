from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from datasets import load_dataset
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_for_eval(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_perplexity_eval(
    model_path: str | Path,
    eval_dataset: str = "wikitext",
    eval_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    model, tokenizer = load_model_for_eval(model_path)

    dataset = load_dataset(eval_dataset, eval_config, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_loss = 0.0
    total_tokens = 0
    max_length = 512

    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            text = dataset[i]["text"]
            if not text or not text.strip():
                continue
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            ).to(model.device)
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"perplexity": perplexity, "loss": avg_loss, "type": "perplexity"}


def run_generation_eval(
    model_path: str | Path,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> list[str]:
    model, tokenizer = load_model_for_eval(model_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=model.dtype,
        device_map="auto",
    )

    outputs = pipe(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    return [out[0]["generated_text"] for out in outputs]


def run_evaluation_harness(
    model_path: str | Path,
    *,
    tasks: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    if tasks is None:
        tasks = ["perplexity"]

    results: dict[str, Any] = {}

    if "perplexity" in tasks:
        try:
            results["perplexity"] = run_perplexity_eval(
                model_path,
                max_samples=limit or 100,
            )
        except Exception as e:
            logger.warning(f"Perplexity eval failed: {e}")
            results["perplexity"] = {"error": str(e)}

    return results
