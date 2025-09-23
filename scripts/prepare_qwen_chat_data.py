#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_SYSTEM = "你是一个任务编排者，你需要根据用户的指令以及可用的专家列表，规划出求解步骤，选出每步合适的专家，并且仅输出json数组。"


def build_prompt(entry: dict, system_prompt: str, tokenizer, enable_thinking: bool) -> str:
    request = (entry.get("input", {}).get("user_request") or "").strip()
    tools = entry.get("input", {}).get("available_tools") or []
    lines = []
    for idx, tool in enumerate(tools, start=1):
        tid = tool.get("id", "")
        name = tool.get("name", "")
        desc = tool.get("description", "")
        lines.append(f"{idx}. {tid}「{name}」：{desc}")
    tools_text = "\n".join(lines)
    user_text = (
        f"{request}\n\n"
        "可用专家列表：\n"
        f"{tools_text}\n\n"
        "请根据这些专家规划执行步骤，并仅输出 JSON 数组，每个元素包含 step_name 与 agent_name 字段，不要添加额外说明。"
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt = build_prompt(entry, system_prompt, tokenizer, enable_thinking)
            response = build_response(entry)
            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Qwen chat-format SFT data from tool selection dataset")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument("--tokenizer", type=str, default="/home/choco/models/qwen3-0.6b")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode in chat template")
    args = parser.parse_args(argv)

    convert_file(args.input, args.output, args.tokenizer, args.system_prompt, args.enable_thinking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
