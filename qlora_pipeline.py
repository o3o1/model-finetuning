#!/usr/bin/env python3
"""End-to-end QLoRA distillation and evaluation pipeline.

This script provides two subcommands:
  - train: finetune a student model with QLoRA on the prepared SFT dataset.
  - evaluate: run the finetuned model on the test split and compute text-similarity metrics.

It assumes the repo's `data/sft/{train,val,test}.jsonl` files already exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


try:
    import evaluate as hf_evaluate
    import sacrebleu
except ImportError as exc:  # pragma: no cover - guidance for missing deps
    raise SystemExit(
        "Missing evaluation dependencies. Run `uv add evaluate sacrebleu rouge-score`."
    ) from exc

try:  # optional progress bars for evaluation
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


DEFAULT_SYSTEM = """你是一个任务编排者，你需要根据用户的指令以及可用的专家列表，规划出求解步骤，选出每步合适的专家，并且仅输出json数组。"""


def resolve_system_prompt(explicit_prompt: Optional[str], data_dir: Path) -> str:
    if explicit_prompt:
        return explicit_prompt
    prompt_file = data_dir / "system_prompt.txt"
    if prompt_file.exists():
        text = prompt_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_SYSTEM


@dataclass
class Prompter:
    tokenizer: AutoTokenizer
    system_prompt: str
    use_chat_template: bool = True
    enable_thinking: bool = False

    def build_input(self, prompt: str) -> str:
        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        sys_part = self.system_prompt.strip()
        if sys_part and not sys_part.endswith("\n\n"):
            sys_part += "\n\n"
        return f"{sys_part}{prompt.strip()}\n\n"

    def build_supervised_text(self, prompt: str, response: str) -> str:
        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
            )
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token and not text.endswith(eos_token):
                text = text + eos_token
            return text
        return self.build_input(prompt) + response.strip()


def load_sft_datasets(data_dir: Path) -> DatasetDict:
    files = {
        split: str(data_dir / f"{split}.jsonl")
        for split in ["train", "val", "test"]
        if (data_dir / f"{split}.jsonl").exists()
    }
    if "train" not in files:
        raise FileNotFoundError("Expected train.jsonl inside data directory")
    ds_dict = load_dataset("json", data_files=files)
    return DatasetDict({k: v for k, v in ds_dict.items()})


def prepare_supervised_dataset(
    dataset: Dataset,
    tokenizer,
    prompter: Prompter,
    add_eos: bool = True,
    keep_columns: Optional[List[str]] = None,
) -> Dataset:
    keep_columns = keep_columns or []

    def _format(example: Dict[str, str]) -> Dict[str, str]:
        prompt_val = example.get("prompt", "")
        response_val = example.get("response", "")

        if prompter.use_chat_template and isinstance(prompt_val, str) and "<|im_start|>" in prompt_val:
            text = prompt_val
            if not text.endswith("\n"):
                text += "\n"
            text += response_val.rstrip()
            if not text.endswith("\n"):
                text += "\n"
            if "<|im_end|>" not in text.split(response_val.rstrip())[-1]:
                text += "<|im_end|>\n"
        else:
            text = prompter.build_supervised_text(prompt_val, response_val)

        if add_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token
        result = {"text": text}
        for col in keep_columns:
            if col in example:
                result[col] = example[col]
        return result

    return dataset.map(_format, remove_columns=[col for col in dataset.column_names if col not in keep_columns])


def get_quantization_config(
    use_4bit: bool,
    *,
    compute_dtype: torch.dtype = torch.float16,
    bnb_nf4: bool = True,
) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4" if bnb_nf4 else "fp4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def prepare_tokenizer(model_name: str, trust_remote_code: bool, *, force_left_padding: bool = False) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if force_left_padding:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"
    return tokenizer


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_dict = load_sft_datasets(data_dir)
    tokenizer = prepare_tokenizer(
        args.model_name,
        args.trust_remote_code,
        force_left_padding=args.left_padding,
    )
    prompter = Prompter(
        tokenizer=tokenizer,
        system_prompt=resolve_system_prompt(args.system_prompt, data_dir),
        use_chat_template=not args.disable_chat_template,
        enable_thinking=args.enable_thinking,
    )

    add_eos = not prompter.use_chat_template
    train_dataset = prepare_supervised_dataset(datasets_dict["train"], tokenizer, prompter, add_eos=add_eos)
    try:
        preview = train_dataset[0]
    except IndexError:
        preview = None
    if preview is not None:
        sample_text = preview.get("text") if isinstance(preview, dict) else preview
        print("[QLoRA] 首条训练样本: ")
        print(sample_text)
    eval_dataset = None
    if "val" in datasets_dict:
        eval_dataset = prepare_supervised_dataset(datasets_dict["val"], tokenizer, prompter, add_eos=add_eos)

    bf16_capable = False
    if torch.cuda.is_available():
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            bf16_capable = bool(checker())
        else:  # fallback based on compute capability (Ampere+)
            major, _ = torch.cuda.get_device_capability()
            bf16_capable = major >= 8

    if args.bf16 and not bf16_capable:
        print("[QLoRA] Requested bf16 but GPU/driver does not support it; falling back to fp16")

    bf16_on = args.bf16 and bf16_capable
    fp16_on = args.fp16 and torch.cuda.is_available() and not bf16_on

    bnb_compute_dtype = torch.bfloat16 if bf16_on else torch.float16
    quant_config = get_quantization_config(
        use_4bit=not args.no_quant,
        compute_dtype=bnb_compute_dtype,
    )
    if quant_config is None:
        print("[QLoRA] Loading base model in full precision (4bit disabled)")
    else:
        print(
            "[QLoRA] Loading base model with 4bit quantization",
            f"(compute dtype={quant_config.bnb_4bit_compute_dtype})",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules or None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    save_strategy = args.save_strategy
    eval_strategy = args.eval_strategy if eval_dataset is not None else "no"

    if args.load_best_model_at_end and eval_dataset is not None and eval_strategy != "no":
        if save_strategy != eval_strategy:
            print(
                "[QLoRA] Aligning eval_strategy to save_strategy for load_best_model_at_end",
                f"({eval_strategy} -> {save_strategy})",
            )
            eval_strategy = save_strategy

    sft_kwargs = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": eval_dataset is not None,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "logging_steps": args.logging_steps,
        "lr_scheduler_type": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optimizer,
        "bf16": bf16_on,
        "fp16": fp16_on,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "save_total_limit": args.save_total_limit,
        "push_to_hub": args.push_to_hub,
        "max_length": args.max_seq_length,
        "dataset_text_field": "text",
    }

    if args.gradient_checkpointing:
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gc_use_reentrant}

    save_steps = args.save_steps
    eval_steps = args.eval_steps if eval_dataset is not None else None

    if save_strategy == "steps":
        if eval_steps is not None and eval_strategy == "steps":
            if save_steps % eval_steps != 0:
                save_steps = ((save_steps // eval_steps) + 1) * eval_steps
        sft_kwargs["save_steps"] = save_steps

    if eval_steps is not None and eval_strategy == "steps":
        sft_kwargs["eval_steps"] = eval_steps

    if eval_dataset is not None and args.load_best_model_at_end:
        sft_kwargs["load_best_model_at_end"] = True
        sft_kwargs["metric_for_best_model"] = args.metric_for_best_model
        sft_kwargs["greater_is_better"] = args.greater_is_better

    # Filter out None to keep defaults from SFTConfig
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if v is not None}
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    resume_arg = getattr(args, "resume_from_checkpoint", None)
    if resume_arg:
        resume_candidate = Path(resume_arg)
        resolved_checkpoint: Optional[str] = None

        if resume_candidate.is_dir():
            if (resume_candidate / "trainer_state.json").exists():
                resolved_checkpoint = str(resume_candidate)
            else:
                last_ckpt = get_last_checkpoint(str(resume_candidate))
                if last_ckpt:
                    resolved_checkpoint = last_ckpt
        elif resume_candidate.exists() and resume_candidate.is_file():
            if resume_candidate.suffix in {".json", ""}:
                try:
                    data = resume_candidate.read_text(encoding="utf-8").strip()
                    if data.startswith("{"):
                        checkpoint_name = json.loads(data).get("last_checkpoint")
                    else:
                        checkpoint_name = data
                    if checkpoint_name:
                        candidate_dir = resume_candidate.parent / checkpoint_name
                        if candidate_dir.is_dir():
                            resolved_checkpoint = str(candidate_dir)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[QLoRA] Failed to parse {resume_candidate}: {exc}")
            # fall through to searching parent if file could not be resolved
        if resolved_checkpoint is None:
            search_root = resume_candidate.parent if resume_candidate.parent.exists() else Path.cwd()
            last_ckpt = get_last_checkpoint(str(search_root))
            if last_ckpt:
                resolved_checkpoint = last_ckpt

        if resolved_checkpoint is None:
            fallback_ckpt = get_last_checkpoint(str(output_dir))
            if fallback_ckpt:
                resolved_checkpoint = fallback_ckpt

        if resolved_checkpoint is None:
            raise FileNotFoundError(
                f"Unable to locate checkpoint for resume path '{resume_arg}'. "
                "Provide a checkpoint directory such as '.../checkpoint-125'."
            )
        print(f"[QLoRA] Resuming from checkpoint: {resolved_checkpoint}")
        trainer.train(resume_from_checkpoint=resolved_checkpoint)
    else:
        trainer.train()

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.merge_lora:
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


def extract_answer(full_text: str) -> str:
    text = full_text.strip()
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "</think>" in text:
        after_think = text.split("</think>", 1)[-1]
        text = after_think
    if "<think>" in text and "</think>" not in text:
        # remove dangling think block
        text = text.split("<think>", 1)[0]
    return text.strip()


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


def generate_predictions(
    model: PeftModelForCausalLM | AutoModelForCausalLM,
    tokenizer,
    dataset: Dataset,
    prompter: Prompter,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    do_sample: bool,
    presence_penalty: Optional[float],
    batch_size: int,
    device: torch.device,
    input_max_length: int,
    *,
    show_progress: bool = False,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    model.eval()
    progress_bar = None
    indices = range(0, len(dataset), batch_size)
    if show_progress and tqdm is not None:
        progress_bar = tqdm(total=len(dataset), desc="Evaluating", leave=False)
    for start in indices:
        raw_batch = dataset[start : start + batch_size]
        if isinstance(raw_batch, dict):
            batch_size_eff = len(next(iter(raw_batch.values()))) if raw_batch else 0
            batch = [
                {key: raw_batch[key][i] for key in raw_batch}
                for i in range(batch_size_eff)
            ]
        else:
            batch = list(raw_batch)
        if not batch:
            continue
        prompts = []
        for item in batch:
            prompt_text = item.get("prompt") or item.get("user_request") or ""
            if isinstance(prompt_text, str) and "<|im_start|>" in prompt_text:
                prompts.append(prompt_text)
            else:
                prompts.append(prompter.build_input(prompt_text))
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if presence_penalty is not None and presence_penalty != 0.0:
            gen_kwargs["repetition_penalty"] = 1.0 + presence_penalty
        with torch.no_grad():
            generation = model.generate(**inputs, **gen_kwargs)
        for idx, output_ids in enumerate(generation):
            prompt_len = int(inputs["attention_mask"][idx].sum().item())
            generated_ids = output_ids[prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            pred = extract_answer(text)
            example = batch[idx]
            results.append(
                {
                    "id": example.get("id", str(start + idx)),
                    "prompt": example["prompt"],
                    "prediction": pred,
                    "reference": example.get("response", ""),
                }
            )
        if progress_bar is not None:
            progress_bar.update(len(batch))
    if progress_bar is not None:
        progress_bar.close()
    return results


def evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    adapter_dir = Path(args.adapter_dir)
    datasets_dict = load_sft_datasets(data_dir)
    test_dataset = datasets_dict.get("test")
    if test_dataset is None:
        raise FileNotFoundError("Expected test.jsonl for evaluation")

    tokenizer_source = adapter_dir if adapter_dir.exists() else Path(args.model_name)
    tokenizer = prepare_tokenizer(str(tokenizer_source), args.trust_remote_code, force_left_padding=not args.right_padding)

    quant_config = get_quantization_config(use_4bit=not args.no_quant)
    if quant_config is None:
        print("[Eval] Loading base model in full precision (4bit disabled)")
    else:
        print(
            "[Eval] Loading base model with 4bit quantization",
            f"(compute dtype={quant_config.bnb_4bit_compute_dtype})",
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    if adapter_dir.exists():
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model

    device = resolve_inference_device(model)

    prompter = Prompter(
        tokenizer=tokenizer,
        system_prompt=resolve_system_prompt(args.system_prompt, data_dir),
        use_chat_template=not args.disable_chat_template,
        enable_thinking=args.enable_thinking,
    )
    generations = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        prompter=prompter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        presence_penalty=args.presence_penalty,
        batch_size=args.batch_size,
        device=device,
        input_max_length=args.input_max_length,
        show_progress=True,
    )

    predictions = [item["prediction"] for item in generations]
    references = [item["reference"] for item in generations]

    if not predictions or not references:
        print(json.dumps({
            "rouge": {},
            "bleu": None,
            "avg_pred_chars": 0,
            "avg_ref_chars": 0,
            "count": 0,
            "note": "No generations available for evaluation"
        }, ensure_ascii=False, indent=2))
        return

    rouge = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    avg_pred_len = sum(len(p) for p in predictions) / max(len(predictions), 1)
    avg_ref_len = sum(len(r) for r in references) / max(len(references), 1)

    metrics = {
        "rouge": rouge_scores,
        "bleu": bleu.score,
        "avg_pred_chars": avg_pred_len,
        "avg_ref_chars": avg_ref_len,
        "count": len(predictions),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_generations:
        out_path = Path(args.save_generations)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in generations:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA distillation + evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Finetune student model with QLoRA")
    train_p.add_argument("--model-name", required=True, help="Base student model identifier")
    train_p.add_argument("--output-dir", default="artifacts/qlora", help="Directory for LoRA adapter outputs")
    train_p.add_argument("--data-dir", default="data/sft", help="Directory containing train/val/test jsonl")
    train_p.add_argument("--system-prompt", help="Optional system prompt override")
    train_p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for custom models")
    train_p.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantized loading")
    train_p.add_argument("--left-padding", action="store_true", help="Use left padding during training inputs")
    train_p.add_argument("--disable-chat-template", action="store_true", help="Disable tokenizer chat template when formatting prompts")
    train_p.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode (default off for this task)")
    train_p.add_argument("--per-device-train-batch-size", type=int, default=2)
    train_p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_p.add_argument("--learning-rate", type=float, default=2e-4)
    train_p.add_argument("--weight-decay", type=float, default=0.0)
    train_p.add_argument("--num-train-epochs", type=float, default=6.0)
    train_p.add_argument("--max-seq-length", type=int, default=4096)
    train_p.add_argument("--optimizer", default="paged_adamw_32bit")
    train_p.add_argument("--lr-scheduler", default="linear")
    train_p.add_argument("--warmup-ratio", type=float, default=0.05)
    train_p.add_argument("--logging-steps", type=int, default=10)
    train_p.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        dest="eval_strategy",
        help="Evaluation schedule (mirrors TrainingArguments eval_strategy)",
    )
    train_p.add_argument("--save-strategy", default="steps")
    train_p.add_argument("--save-steps", type=int, default=100)
    train_p.add_argument("--eval-steps", type=int, default=100)
    train_p.add_argument("--load-best-model-at-end", action="store_true")
    train_p.add_argument("--metric-for-best-model", default="eval_loss")
    train_p.add_argument("--greater-is-better", action="store_true")
    train_p.add_argument("--save-total-limit", type=int, help="Maximum number of checkpoints to keep")
    train_p.add_argument("--push-to-hub", action="store_true")
    train_p.add_argument("--lora-r", type=int, default=32)
    train_p.add_argument("--lora-alpha", type=int, default=64)
    train_p.add_argument("--lora-dropout", type=float, default=0.1)
    train_p.add_argument("--lora-target-modules", nargs="+", help="Restrict LoRA to specific target modules")
    train_p.add_argument("--gradient-checkpointing", action="store_true")
    train_p.add_argument(
        "--gc-use-reentrant",
        action="store_true",
        help="Use reentrant checkpointing (default False for Torch>=2.5 recommendation)",
    )
    train_p.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume training from")
    train_p.add_argument("--bf16", action="store_true")
    train_p.add_argument("--fp16", action="store_true")
    train_p.add_argument("--report-to", nargs="+", help="Optional trackers e.g. wandb")
    train_p.add_argument("--merge-lora", action="store_true", help="Merge LoRA adapter into base weights after training")
    train_p.set_defaults(func=train)

    eval_p = sub.add_parser("evaluate", help="Evaluate the finetuned adapter")
    eval_p.add_argument("--model-name", required=True, help="Base model identifier used for finetuning")
    eval_p.add_argument("--adapter-dir", default="artifacts/qlora", help="Directory where LoRA adapter is stored")
    eval_p.add_argument("--data-dir", default="data/sft", help="Directory with test split jsonl")
    eval_p.add_argument("--system-prompt", help="Optional system prompt override")
    eval_p.add_argument("--trust-remote-code", action="store_true")
    eval_p.add_argument("--no-quant", action="store_true")
    eval_p.add_argument("--right-padding", action="store_true", help="Use right padding during generation (default left)")
    eval_p.add_argument("--disable-chat-template", action="store_true")
    eval_p.add_argument("--enable-thinking", action="store_true")
    eval_p.add_argument("--max-new-tokens", type=int, default=2048)
    eval_p.add_argument("--temperature", type=float, default=0.7)
    eval_p.add_argument("--top-p", type=float, default=0.8)
    eval_p.add_argument("--top-k", type=int, default=20)
    eval_p.add_argument("--no-sample", action="store_true", help="Disable sampling (defaults to Qwen recommended sampling)")
    eval_p.add_argument("--batch-size", type=int, default=2)
    eval_p.add_argument("--input-max-length", type=int, default=1024)
    eval_p.add_argument("--save-generations", help="Optional path to save prediction/reference pairs as JSONL")
    eval_p.add_argument("--presence-penalty", type=float, default=0.0)
    eval_p.set_defaults(func=evaluate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
