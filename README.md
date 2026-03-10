# LLM Post-Training Pipeline

A modular pipeline for LLM alignment: supervised fine-tuning (SFT), reward model training, and direct preference optimization (DPO). Implements the standard RLHF-style workflow used in model alignment, scaled down to run on a laptop (M2) or single GPU. Learning project to experiment with alignment techniques.

## Overview

The pipeline follows the alignment stack: SFT on instruction data → optional reward model on preferences → DPO for preference optimization. Each stage is a separate component with Hydra configs, so you can run them independently or chain them.

```
Dataset → SFT → [Reward Model] → DPO → Evaluation
```

Built with HuggingFace TRL, PEFT (LoRA/QLoRA), and Hydra. W&B for experiment tracking when enabled.

## Setup

```bash
uv sync
```

Requires [uv](https://docs.astral.sh/uv/). Python 3.11+.

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` (HuggingFace), `WANDB_MODE=disabled` (skip W&B), or cache paths.

## Quick Start

SFT on the example dataset (8 samples, ~30s on M2):

```bash
WANDB_MODE=disabled uv run python scripts/run_training.py stage=sft_m2 dataset=alpaca_local
```

Compare baseline vs fine-tuned:

```bash
uv run python scripts/compare_models.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 outputs/sft/final --max-samples 20
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **SFT** | Instruction tuning with TRL SFTTrainer. LoRA or QLoRA. |
| **Reward model** | Bradley-Terry preference learning with TRL RewardTrainer. |
| **DPO** | Direct preference optimization (no separate reward model). |
| **Eval** | Perplexity on Wikitext, side-by-side generation comparison. |

## Configuration

Configs are in `configs/`. Override from CLI:

```bash
uv run post-train stage=sft dataset=alpaca_local dataset.max_samples=100
uv run post-train stage=dpo sft_checkpoint=outputs/sft/final
```

**Models:** `model=0.5b` (TinyLlama + LoRA), `model=1b_qlora` (4-bit for low memory)

**Stages:** `sft`, `sft_m2`, `reward_model`, `reward_model_m2`, `dpo`, `dpo_m2` — `_m2` variants use smaller batches for laptops

**Datasets:** `alpaca`, `alpaca_local`, `preference`, `preference_local`

## Data Formats

**SFT:** Alpaca-style `instruction`, `input`, `output`

**Preference (reward model / DPO):** `prompt`, `chosen`, `rejected`

```bash
# Download Alpaca
uv run python scripts/prepare_dataset.py alpaca --output datasets/alpaca --max-samples 1000

# Synthetic preference pairs (for testing only)
uv run python scripts/prepare_dataset.py preference-pairs --input datasets/alpaca_sample.json
```

## Low-Memory / M2

The pipeline detects Apple Silicon or missing CUDA and reduces batch size. Use `model=1b_qlora` for 4-bit training on M2. Set `dataset.max_samples` to cap dataset size for quick runs.

## Project Structure

```
post_training_pipeline/
├── configs/          # Hydra configs (model, stage, dataset)
├── datasets/         # Loaders and preprocessing
├── models/           # Model loading (LoRA, QLoRA)
├── pipelines/        # SFT, reward model, DPO
├── evaluation/       # Perplexity and model comparison
├── scripts/          # prepare_dataset, run_training, compare_models
└── tests/
```

## Development

```bash
uv run pytest
uv run ruff check .
```

Pre-commit hook (ruff check + format):

```bash
uv sync && uv run pre-commit install
```
