#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and export to GGUF via llama.cpp."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def merge_lora(base_model: str, adapter_dir: str, output_dir: Path) -> Path:
    print(f"[merge] base={base_model} adapter={adapter_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer_src = Path(adapter_dir)
    if not (tokenizer_src / "tokenizer_config.json").exists():
        tokenizer_src = Path(base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge] merged model saved at {output_dir}")
    return output_dir


def export_to_gguf(
    merged_dir: Path,
    llama_cpp_path: Path,
    gguf_out: Path,
    quant: str | None,
    llama_python: Path,
    outtype: str,
) -> Path:
    convert_py = llama_cpp_path / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_cpp_path}")

    if gguf_out.suffix:
        gguf_stem = gguf_out.with_suffix("")
    else:
        gguf_stem = gguf_out
    cmd = [
        str(llama_python),
        str(convert_py),
        str(merged_dir),
        "--outfile",
        str(gguf_stem),
        "--outtype",
        outtype,
    ]
    run(cmd)

    explicit_target = gguf_out if gguf_out.suffix == ".gguf" else None
    candidates = sorted(
        gguf_stem.parent.glob("*.gguf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # convert.py default: <outfile> when suffix missing, or <outfile>-<outtype>.gguf
    if explicit_target and explicit_target.exists():
        f16_path = explicit_target
    elif (gguf_stem.parent / gguf_stem.name).is_file():
        src = gguf_stem.parent / gguf_stem.name
        if explicit_target:
            src = src.rename(explicit_target)
            f16_path = explicit_target
        else:
            f16_path = src
    elif candidates:
        f16_path = candidates[0]
        if explicit_target and not explicit_target.exists():
            f16_path = f16_path.rename(explicit_target)
        elif explicit_target and explicit_target.exists():
            f16_path = explicit_target
    else:
        raise FileNotFoundError("Failed to locate generated GGUF file")

    if quant:
        quant_bin_candidates = [
            llama_cpp_path / "build" / "bin" / "llama-quantize",
            llama_cpp_path / "llama-quantize",
            llama_cpp_path / "build" / "bin" / "quantize",
        ]
        quant_bin = next((p for p in quant_bin_candidates if p.exists()), None)
        if quant_bin is None:
            raise FileNotFoundError("quantize binary not found under llama.cpp build")
        quant_out = gguf_out if gguf_out.suffix == ".gguf" else gguf_stem.with_name(f"{gguf_stem.name}-{quant}.gguf")
        run([str(quant_bin), str(f16_path), str(quant_out), quant])
        print(f"[quant] generated {quant_out} with method {quant}")
        return quant_out
    return f16_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF using llama.cpp")
    ap.add_argument("--base-model", required=True, help="Base student model path or HF id")
    ap.add_argument("--adapter", required=True, help="Trained LoRA adapter directory")
    ap.add_argument("--llama-cpp", required=True, help="Path to llama.cpp repository root")
    ap.add_argument("--output", default="artifacts/export", help="Output directory for merged model/GGUF")
    ap.add_argument("--gguf-name", default="finetuned", help="Base filename for GGUF (without extension)")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], help="Output precision passed to convert_hf_to_gguf.py")
    ap.add_argument("--quant", help="Optional quantization target (e.g., Q4_0, Q5_K_M)")
    ap.add_argument("--llama-python", help="Path to python executable inside llama.cpp virtualenv")
    args = ap.parse_args()

    output_dir = Path(args.output)
    merged_dir = merge_lora(args.base_model, args.adapter, output_dir / "merged")

    llama_python = Path(args.llama_python) if args.llama_python else Path(sys.executable)
    gguf_base = output_dir / args.gguf_name
    gguf_path = export_to_gguf(merged_dir, Path(args.llama_cpp), gguf_base, args.quant, llama_python, args.outtype)
    print(f"[done] GGUF available at {gguf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
