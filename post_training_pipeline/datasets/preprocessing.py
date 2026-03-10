from typing import Any

from transformers import PreTrainedTokenizer

from datasets import Dataset
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def format_sft_example(
    example: dict[str, Any],
    *,
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    template: str | None = None,
) -> dict[str, Any]:
    instruction = example.get(instruction_key, "")
    inp = example.get(input_key, "")
    output = example.get(output_key, "")

    if inp:
        instruction = f"{instruction}\n\nInput: {inp}" if instruction else inp

    if template:
        text = template.format(instruction=instruction, response=output)
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": text, "instruction": instruction, "output": output}


def format_preference_example(
    example: dict[str, Any],
    *,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> dict[str, Any]:
    prompt = example.get(prompt_key) or example.get("instruction", "")
    chosen = example.get(chosen_key, "")
    rejected = example.get(rejected_key, "")

    if not prompt or not chosen or not rejected:
        raise ValueError("Preference example must have prompt, chosen, and rejected")

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def apply_chat_template_to_preference(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    *,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> dict[str, Any]:
    prompt = example[prompt_key]
    chosen = example[chosen_key]
    rejected = example[rejected_key]

    chosen_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected},
    ]

    return {
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "chosen": tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
        "rejected": tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    stage: str,
    dataset_config: dict[str, Any],
) -> Dataset:
    if stage == "sft":
        return _preprocess_sft(dataset, tokenizer, dataset_config)
    if stage in ("reward_model", "dpo"):
        return _preprocess_preference(dataset, tokenizer, dataset_config)
    raise ValueError(f"Unknown stage: {stage}")


def _preprocess_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
) -> Dataset:
    text_column = config.get("text_column", "text")
    instruction_key = config.get("instruction_key", "instruction")
    input_key = config.get("input_key", "input")
    output_key = config.get("output_key", "output")

    def format_fn(ex: dict[str, Any]) -> dict[str, Any]:
        return format_sft_example(
            ex,
            instruction_key=instruction_key,
            input_key=input_key,
            output_key=output_key,
        )

    if text_column not in dataset.column_names:
        dataset = dataset.map(format_fn, remove_columns=dataset.column_names, desc="Format SFT")

    return dataset


def _preprocess_preference(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: dict[str, Any],
) -> Dataset:
    prompt_key = config.get("prompt_key", "prompt")
    chosen_key = config.get("chosen_key", "chosen")
    rejected_key = config.get("rejected_key", "rejected")
    use_chat_template = config.get("use_chat_template", True)

    def format_fn(ex: dict[str, Any]) -> dict[str, Any]:
        out = format_preference_example(
            ex,
            prompt_key=prompt_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
        )
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                out = apply_chat_template_to_preference(
                    out,
                    tokenizer,
                    prompt_key="prompt",
                    chosen_key="chosen",
                    rejected_key="rejected",
                )
            except Exception as e:
                logger.debug("Chat template failed: %s", e)
        return out

    return dataset.map(format_fn, desc="Format preference", remove_columns=dataset.column_names)
