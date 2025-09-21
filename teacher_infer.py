#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
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


class RateLimiter:
    def __init__(self, rpm: Optional[int], tpm: Optional[int]) -> None:
        self.rpm = rpm or 0
        self.tpm = tpm or 0
        self._req_times = deque()
        self._token_window = deque()
        self._token_sum = 0

    def _purge(self, now: float) -> None:
        cutoff = now - 60.0
        while self._req_times and self._req_times[0] <= cutoff:
            self._req_times.popleft()
        while self._token_window and self._token_window[0][0] <= cutoff:
            _, used = self._token_window.popleft()
            self._token_sum -= used

    def acquire(self) -> None:
        if not self.rpm:
            return
        while True:
            now = time.monotonic()
            self._purge(now)
            if len(self._req_times) < self.rpm:
                self._req_times.append(now)
                return
            wait = self._req_times[0] + 60.0 - now
            if wait > 0:
                time.sleep(wait)
            else:
                self._req_times.popleft()

    def note_tokens(self, tokens: Optional[int]) -> None:
        if not self.tpm or tokens is None:
            return
        remaining = tokens
        while remaining > 0:
            now = time.monotonic()
            self._purge(now)
            available = self.tpm - self._token_sum
            if available > 0:
                take = min(remaining, available)
                self._token_window.append((now, take))
                self._token_sum += take
                remaining -= take
                continue
            wait = self._token_window[0][0] + 60.0 - now
            if wait > 0:
                time.sleep(wait)
            else:
                _, used = self._token_window.popleft()
                self._token_sum -= used


def call_teacher(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> tuple[str, Optional[int]]:
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
    content = message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = (content or "") if isinstance(content, str) else ""
    text = text.strip()
    if not text:
        raise RuntimeError("Empty response content from teacher model")
    usage = getattr(response, "usage", None)
    total_tokens: Optional[int] = None
    if usage is not None:
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens") or usage.get("total_tokens_used")
        else:
            total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "total_tokens_used", None)
    return text, total_tokens


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
    rpm: Optional[int],
    tpm: Optional[int],
    limit: Optional[int],
) -> None:
    results: list[Dict[str, object]] = []
    total = 0
    existing_ids: set[str] = set()
    if output_path.exists():
        for row in read_jsonl(output_path):
            rid = row.get("id")
            if isinstance(rid, str):
                existing_ids.add(rid)
        if existing_ids:
            print(f"[info] Found {len(existing_ids)} existing responses, skipping duplicates", file=sys.stderr)
    limiter = RateLimiter(rpm=rpm, tpm=tpm)
    for seed in read_jsonl(seeds_path):
        seed_id = seed.get("id")
        prompt = (seed.get("prompt") or "").strip()
        if not seed_id or not prompt:
            continue
        if seed_id in existing_ids:
            continue
        attempt = 0
        while True:
            try:
                limiter.acquire()
                answer, tokens_used = call_teacher(
                    client,
                    model,
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                limiter.note_tokens(tokens_used)
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
        existing_ids.add(seed_id)
        if flush_every and len(results) >= flush_every:
            append_jsonl(output_path, results)
            results.clear()
            print(f"[flush] wrote {total} items", file=sys.stderr)
            time.sleep(retry_delay)
        if limit and total >= limit:
            break
    if results:
        append_jsonl(output_path, results)
        print(f"[flush] wrote {total} items", file=sys.stderr)


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call OpenAI-compatible teacher model using seed prompts")
    parser.add_argument("--input", type=Path, default=Path("data/seed/seed_prompts.jsonl"), help="Seed prompts JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/teacher_outputs/teacher.jsonl"), help="Where to store teacher responses")
    parser.add_argument("--model", default="qwen3-next-80b-a3b-instruct", help="Teacher model name")
    parser.add_argument("--api-key", dest="api_key", help="Override OPENAI_API_KEY")
    parser.add_argument(
        "--api-base",
        dest="api_base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Override API base URL",
    )
    parser.add_argument("--system", dest="system_prompt", help="Optional system prompt for the teacher model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int)
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=3)
    parser.add_argument("--retry-delay", dest="retry_delay", type=float, default=2.0, help="Seconds to sleep between retries")
    parser.add_argument("--flush-every", dest="flush_every", type=int, default=20, help="Flush output to disk every N samples")
    parser.add_argument("--rpm", type=int, default=600, help="Requests per minute budget")
    parser.add_argument("--tpm", type=int, default=1_000_000, help="Tokens per minute budget")
    parser.add_argument("--limit", type=int, help="Limit number of prompts to process")
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
        rpm=ns.rpm,
        tpm=ns.tpm,
        limit=ns.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
