#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def prepare_alpaca(
    output_dir: Path,
    max_samples: int | None = None,
) -> None:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alpaca_train.json"
    ds.to_json(output_path)
    print(f"Saved {len(ds)} samples to {output_path}")


def convert_to_alpaca(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    else:
        items = [data]

    def map_item(item: dict) -> dict:
        return {
            "instruction": item.get("instruction", item.get("prompt", item.get("question", ""))),
            "input": item.get("input", item.get("context", "")),
            "output": item.get("output", item.get("response", item.get("answer", ""))),
        }

    converted = [map_item(item) for item in items]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)
    print(f"Converted {len(converted)} samples to {output_path}")


def create_preference_pairs(
    input_path: Path,
    output_path: Path,
    num_pairs: int = 50,
) -> None:
    with open(input_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    else:
        items = [data]

    pairs = []
    reject_templates = [
        "I'm not sure.",
        "That's a difficult question.",
        "I don't have enough information to answer.",
    ]

    for i, item in enumerate(items[:num_pairs]):
        instruction = item.get("instruction", item.get("prompt", ""))
        inp = item.get("input", "")
        chosen = item.get("output", item.get("response", ""))
        prompt = f"{instruction}\n\nInput: {inp}" if inp else instruction
        rejected = reject_templates[i % len(reject_templates)]
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Created {len(pairs)} preference pairs at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_alpaca = subparsers.add_parser("alpaca", help="Download Alpaca dataset")
    p_alpaca.add_argument("--output", type=Path, default=Path("datasets/alpaca"))
    p_alpaca.add_argument("--max-samples", type=int, default=None)

    p_convert = subparsers.add_parser("convert", help="Convert to Alpaca format")
    p_convert.add_argument("--input", type=Path, required=True)
    p_convert.add_argument("--output", type=Path, required=True)

    p_pref = subparsers.add_parser("preference-pairs", help="Create synthetic preference pairs")
    p_pref.add_argument("--input", type=Path, required=True)
    p_pref.add_argument("--output", type=Path, default=Path("datasets/preference_synthetic.jsonl"))
    p_pref.add_argument("--num-pairs", type=int, default=50)

    args = parser.parse_args()

    if args.command == "alpaca":
        prepare_alpaca(args.output, args.max_samples)
    elif args.command == "convert":
        convert_to_alpaca(args.input, args.output)
    elif args.command == "preference-pairs":
        create_preference_pairs(args.input, args.output, args.num_pairs)


if __name__ == "__main__":
    main()
