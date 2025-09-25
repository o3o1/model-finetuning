#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_SYSTEM = (
    "你是一名电路拓扑规划师。给定原始网表后，需要识别其中的层级模块，"
    "并仅输出包含 `modules` 数组和 `replaced_netlist` 字段的 JSON 对象。"
)


def build_prompt(entry: dict, system_prompt: str, tokenizer, enable_thinking: bool) -> str:
    netlist = entry.get("input", {}).get("netlist") or []
    lines = [f"{idx}. {line}" for idx, line in enumerate(netlist, start=1)]
    user_text = (
        "原始网表如下:\n"
        + "\n".join(lines)
        + "\n\n请提取模块并仅输出 JSON 对象，包含字段 `modules` 与 `replaced_netlist`。不要添加额外说明。"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_response(entry: dict) -> str:
    output = entry.get("output")
    return json.dumps(output, ensure_ascii=False)


def convert_file(src: Path, dst: Path, tokenizer_name: str, system_prompt: str, enable_thinking: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompt = build_prompt(entry, system_prompt, tokenizer, enable_thinking)
            response = build_response(entry)
            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare chat-format data for netlist module extraction")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default="/home/choco/models/qwen3-0.6b")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)

    convert_file(args.input, args.output, args.tokenizer, args.system_prompt, args.enable_thinking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
