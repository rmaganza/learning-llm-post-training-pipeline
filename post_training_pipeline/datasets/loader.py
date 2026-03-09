from pathlib import Path
from typing import Any, Optional

from datasets.arrow_dataset import Dataset as HFDataset

from datasets import load_dataset
from post_training_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset_from_config(
    name: str,
    split: str = "train",
    path: Optional[str] = None,
    subset: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> HFDataset:
    """Load from HuggingFace Hub. path/name = dataset id, subset = config."""
    try:
        ds = load_dataset(
            path or name,
            name=subset,
            split=split,
            trust_remote_code=trust_remote_code,
            streaming=streaming,
            **kwargs,
        )
        logger.info(f"Loaded dataset: {name}, split={split}, rows={len(ds)}")
        return ds
    except Exception as e:
        logger.error(f"Failed to load dataset {name}: {e}")
        raise


def load_local_dataset(
    data_path: str | Path,
    split: str = "train",
    file_type: Optional[str] = None,
) -> HFDataset:
    """Load from local JSON/JSONL/CSV/Parquet. Handles single file or directory."""
    path = Path(data_path)

    if path.is_dir():
        split_dir = path / split
        if split_dir.exists():
            path = split_dir
        files = (
            list(path.glob("*.json"))
            + list(path.glob("*.jsonl"))
            + list(path.glob("*.csv"))
            + list(path.glob("*.parquet"))
        )
        if not files:
            raise FileNotFoundError(f"No data files found in {path}")
        data_files = str(files[0]) if len(files) == 1 else {split: [str(f) for f in files]}
    else:
        data_files = str(path)
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")

    if file_type is None and isinstance(data_files, str):
        suffix = Path(data_files).suffix.lower()
        file_type = {"json": "json", "jsonl": "json"}.get(suffix.lstrip("."), suffix.lstrip("."))

    ds = load_dataset(
        "json" if file_type in ("json", "jsonl") else file_type or "json",
        data_files=data_files,
        split=split if isinstance(data_files, dict) else "train",
    )
    logger.info(f"Loaded local dataset: {data_path}, rows={len(ds)}")
    return ds


def get_dataset(
    dataset_config: dict[str, Any],
    split: str = "train",
) -> HFDataset:
    """Load dataset from config. Supports Hub, local files, max_samples, streaming."""
    if dataset_config.get("local", False):
        ds = load_local_dataset(
            dataset_config["name"],
            split=split,
            file_type=dataset_config.get("file_type"),
        )
    else:
        ds = load_dataset_from_config(
            name=dataset_config["name"],
            split=split,
            path=dataset_config.get("path"),
            subset=dataset_config.get("subset"),
            streaming=dataset_config.get("streaming", False),
            trust_remote_code=dataset_config.get("trust_remote_code", False),
        )

    max_samples = dataset_config.get("max_samples")
    if max_samples is not None and max_samples > 0:
        try:
            n = len(ds)
            ds = ds.select(range(min(max_samples, n)))
            logger.info(f"Limited dataset to {max_samples} samples")
        except TypeError:
            # Streaming datasets don't support len()/select; skip limit
            logger.warning("max_samples ignored for streaming dataset")

    return ds
