#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install the openai package (pip install openai)") from exc


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_client(api_key: Optional[str], api_base: Optional[str]) -> OpenAI:
    kwargs: Dict[str, Optional[str]] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def call_teacher(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    text = (message.content or "").strip()
    if not text:
        raise RuntimeError("Empty response content from teacher model")
    return text


def infer(
    seeds_path: Path,
    output_path: Path,
    *,
    client: OpenAI,
    model: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    max_retries: int,
    retry_delay: float,
    flush_every: int,
) -> None:
    results: list[Dict[str, object]] = []
    total = 0
    for seed in read_jsonl(seeds_path):
        seed_id = seed.get("id")
        prompt = (seed.get("prompt") or "").strip()
        if not seed_id or not prompt:
            continue
        attempt = 0
        while True:
            try:
                answer = call_teacher(
                    client,
                    model,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    print(f"[skip] id={seed_id} error={exc}", file=sys.stderr)
                    answer = ""
                    break
                print(f"[retry] id={seed_id} attempt={attempt} error={exc}", file=sys.stderr)
                time.sleep(retry_delay)
        if not answer:
            continue
        results.append({"id": seed_id, "response": answer})
        total += 1
        if flush_every and len(results) >= flush_every:
            append_jsonl(output_path, results)
            results.clear()
            print(f"[flush] wrote {total} items", file=sys.stderr)
            time.sleep(retry_delay)
    if results:
        append_jsonl(output_path, results)
        print(f"[flush] wrote {total} items", file=sys.stderr)


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call OpenAI-compatible teacher model using seed prompts")
    parser.add_argument("--input", type=Path, default=Path("data/seed/seed_prompts.jsonl"), help="Seed prompts JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/teacher_outputs/teacher.jsonl"), help="Where to store teacher responses")
    parser.add_argument("--model", required=True, help="Teacher model name, e.g. gpt-4o-mini")
    parser.add_argument("--api-key", dest="api_key", help="Override OPENAI_API_KEY")
    parser.add_argument("--api-base", dest="api_base", help="Override OPENAI_API_BASE/base URL")
    parser.add_argument("--system", dest="system_prompt", help="Optional system prompt for the teacher model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int)
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=3)
    parser.add_argument("--retry-delay", dest="retry_delay", type=float, default=2.0, help="Seconds to sleep between retries")
    parser.add_argument("--flush-every", dest="flush_every", type=int, default=20, help="Flush output to disk every N samples")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    ns = parse_args(argv)
    client = make_client(ns.api_key, ns.api_base)
    if ns.output.exists():
        print(f"[warn] {ns.output} exists; new outputs will be appended", file=sys.stderr)
    infer(
        ns.input,
        ns.output,
        client=client,
        model=ns.model,
        system_prompt=ns.system_prompt,
        temperature=ns.temperature,
        max_tokens=ns.max_tokens,
        max_retries=ns.max_retries,
        retry_delay=ns.retry_delay,
        flush_every=ns.flush_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
