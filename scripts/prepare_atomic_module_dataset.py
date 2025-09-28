#!/usr/bin/env python3
"""Convert the atomic module dataset into the SFT format used by qlora_pipeline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DEFAULT_PROMPT_TEMPLATE = (
    "原始网表如下:\n{netlist}\n\n"
    "请识别网表中的模块，并仅输出一个 JSON 对象，包含字段 `modules`。"
    "不要添加额外说明。"
)


@dataclass
class SplitConfig:
    train_ratio: float = 0.9
    val_ratio: float = 0.05

    def as_counts(self, total: int) -> tuple[int, int, int]:
        train = int(total * self.train_ratio)
        val = int(total * self.val_ratio)
        test = total - train - val
        if test < 0:
            test = 0
        return train, val, test


def format_prompt(netlist: Iterable[str]) -> str:
    lines = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(netlist))
    return DEFAULT_PROMPT_TEMPLATE.format(netlist=lines)


def convert_record(record: dict) -> dict:
    prompt = format_prompt(record["input"]["netlist"])
    response = json.dumps({"modules": record["output"].get("modules", [])}, ensure_ascii=False)
    return {"prompt": prompt, "response": response}


def load_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/atomic-module/atomic_modules_dataset.jsonl"),
        help="Source JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sft-atomic-module"),
        help="Destination directory for SFT-formatted JSONL splits",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of examples assigned to the training split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of examples assigned to the validation split",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data = load_dataset(args.input)
    random.Random(args.seed).shuffle(data)

    split_cfg = SplitConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_count, val_count, test_count = split_cfg.as_counts(len(data))

    converted = [convert_record(item) for item in data]

    train_rows = converted[:train_count]
    val_rows = converted[train_count : train_count + val_count]
    test_rows = converted[train_count + val_count : train_count + val_count + test_count]

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

