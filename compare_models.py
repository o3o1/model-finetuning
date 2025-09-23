#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DEFAULT_SYSTEM = """你是一名严谨的电力电子专家，请给出准确、结构化的回答。"""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare responses of base and LoRA-adapted models")
    parser.add_argument("--model-name", required=True, help="Base model identifier or local path")
    parser.add_argument("--prompt", help="Inline prompt text")
    parser.add_argument("--prompt-file", type=Path, help="Path to file containing prompt text")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM, help="Optional system prompt override")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-quant", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--left-padding", action="store_true", help="Use left padding for generation")
    parser.add_argument("--disable-chat-template", action="store_true", help="Disable tokenizer.apply_chat_template; use raw prompt")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode when using chat template")
    return parser


def get_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    raise SystemExit("Provide --prompt or --prompt-file")


def format_prompt(
    prompt: str,
    system_prompt: str,
    *,
    disable_chat_template: bool,
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
) -> str:
    if disable_chat_template:
        sys_part = system_prompt.strip()
        if sys_part:
            return f"{sys_part}\n\n{prompt.strip()}"
        return prompt.strip()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def get_quant_config(use_4bit: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def ensure_tokenizer(model_name: str, trust_remote_code: bool, *, left_padding: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if left_padding else "right"
    return tokenizer


def resolve_inference_device(model) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device
    if hasattr(model, "devices") and model.devices:
        return model.devices[0]
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first = next(iter(device_map.values()))
        if isinstance(first, str):
            return torch.device(first)
        if isinstance(first, list) and first:
            return torch.device(first[0])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, tokenizer, prompt_text: str, args: argparse.Namespace) -> str:
    model.eval()
    device = resolve_inference_device(model)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = outputs[0]
    prompt_len = int(inputs["attention_mask"][0].sum().item())
    generated = output_ids[prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def load_base_model(args: argparse.Namespace):
    quant_config = get_quant_config(use_4bit=not args.no_quant)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    return model


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    prompt_raw = get_prompt_text(args)
    tokenizer_source = args.model_name
    if args.adapter and (Path(args.adapter) / "tokenizer_config.json").exists():
        tokenizer_source = args.adapter
    tokenizer = ensure_tokenizer(tokenizer_source, args.trust_remote_code, left_padding=args.left_padding)
    prompt_text = format_prompt(
        prompt_raw,
        args.system_prompt,
        disable_chat_template=args.disable_chat_template,
        tokenizer=tokenizer,
        enable_thinking=args.enable_thinking,
    )

    print("[base] loading model ...")
    base_model = load_base_model(args)
    base_output = generate(base_model, tokenizer, prompt_text, args)
    print("\n===== Base Model Output =====\n")
    print(base_output)

    if args.adapter:
        print("\n[adapter] applying LoRA ...")
        tuned_model = PeftModel.from_pretrained(base_model, args.adapter)
        tuned_output = generate(tuned_model, tokenizer, prompt_text, args)
        print("\n===== LoRA Adapter Output =====\n")
        print(tuned_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
