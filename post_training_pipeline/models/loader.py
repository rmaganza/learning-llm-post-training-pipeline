from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def get_dtype(torch_dtype_str: str) -> torch.dtype:
    """Map string to torch dtype (bfloat16, float16, float32)."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(torch_dtype_str, torch.bfloat16)


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    attn_implementation: str = "sdpa",
    use_flash_attention_2: bool = False,
    gradient_checkpointing: bool = True,
    lora_config: Optional[dict[str, Any]] = None,
    quantization_config: Optional[dict[str, Any]] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load causal LM with optional LoRA and 4/8-bit quantization."""
    dtype = get_dtype(torch_dtype)

    bnb_config = None
    if quantization_config and quantization_config.get("enabled"):
        load_in_4bit = quantization_config.get("load_in_4bit", False)
        load_in_8bit = quantization_config.get("load_in_8bit", False)
        if load_in_4bit or load_in_8bit:
            compute_dtype = get_dtype(quantization_config.get("bnb_4bit_compute_dtype", "bfloat16"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quantization_config.get(
                    "bnb_4bit_use_double_quant", True
                ),
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_kwargs: dict[str, Any] = {}
    if use_flash_attention_2:
        attn_kwargs["attn_implementation"] = "flash_attention_2"
    elif attn_implementation and attn_implementation != "sdpa":
        attn_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb_config,
        **attn_kwargs,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if lora_config and lora_config.get("enabled"):
        target_modules = lora_config.get("target_modules")
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        target_modules = (
            list(target_modules) if target_modules else ["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
        )

        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info("LoRA applied successfully")

    return model, tokenizer


def load_reward_model(
    model_name_or_path: str,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    return load_model_and_tokenizer(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        gradient_checkpointing=True,
        lora_config=None,  # Reward models often trained full or with LoRA
        quantization_config=None,
    )
