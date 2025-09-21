from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# Workspace paths
ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED_DIR = DATA / "seed"
TEACHER_DIR = DATA / "teacher_outputs"
SFT_DIR = DATA / "sft"
EVAL_DIR = DATA / "eval"


# -----------------------------
# Utilities
# -----------------------------


def ensure_dirs() -> None:
    for p in [DATA, SEED_DIR, TEACHER_DIR, SFT_DIR, EVAL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# Domain taxonomy (Power Electronics)
# -----------------------------


TOPICS = {
    "converters": [
        "buck",
        "boost",
        "buck-boost",
        "flyback",
        "forward",
        "half-bridge",
        "full-bridge",
        "LLC",
        "PFC",
    ],
    "control": [
        "voltage-mode",
        "current-mode",
        "PID",
        "digital-control",
        "compensation",
        "soft-start",
    ],
    "devices": [
        "MOSFET",
        "IGBT",
        "SiC",
        "GaN",
        "diode",
        "gate-driver",
    ],
    "magnetics": ["inductor", "transformer", "core", "winding", "saturation"],
    "thermal": ["heat-sink", "losses", "SOA", "Rth", "lifetime"],
    "emi_emc": ["layout", "filter", "common-mode", "differential-mode", "standards"],
    "grid_motor": ["inverter", "rectifier", "motor-drive", "SPWM", "SVPWM"],
}

TOPIC_LABELS = {
    "converters": "电力变换器拓扑",
    "control": "控制策略",
    "devices": "功率器件",
    "magnetics": "磁性元件设计",
    "thermal": "热设计与可靠性",
    "emi_emc": "EMI/EMC 控制",
    "grid_motor": "电网与电机应用",
}

SUBTOPIC_LABELS = {
    "converters": {
        "buck": "Buck 降压变换器",
        "boost": "Boost 升压变换器",
        "buck-boost": "Buck-Boost 升降压变换器",
        "flyback": "反激式变换器",
        "forward": "正激式变换器",
        "half-bridge": "半桥拓扑",
        "full-bridge": "全桥拓扑",
        "LLC": "LLC 谐振变换器",
        "PFC": "功率因数校正（PFC）",
    },
    "control": {
        "voltage-mode": "电压模式控制",
        "current-mode": "电流模式控制",
        "PID": "PID 调节器",
        "digital-control": "数字控制策略",
        "compensation": "补偿网络设计",
        "soft-start": "软启动方案",
    },
    "devices": {
        "MOSFET": "MOSFET 功率晶体管",
        "IGBT": "IGBT 功率器件",
        "SiC": "碳化硅器件 (SiC)",
        "GaN": "氮化镓器件 (GaN)",
        "diode": "功率二极管",
        "gate-driver": "栅极驱动器",
    },
    "magnetics": {
        "inductor": "电感器设计",
        "transformer": "变压器设计",
        "core": "磁芯选型",
        "winding": "绕组布局",
        "saturation": "磁饱和分析",
    },
    "thermal": {
        "heat-sink": "散热器设计",
        "losses": "损耗评估",
        "SOA": "安全工作区 (SOA)",
        "Rth": "热阻 (Rth) 评估",
        "lifetime": "寿命预测",
    },
    "emi_emc": {
        "layout": "布局布线 (Layout)",
        "filter": "滤波器设计",
        "common-mode": "共模干扰",
        "differential-mode": "差模干扰",
        "standards": "EMC 标准符合性",
    },
    "grid_motor": {
        "inverter": "逆变器系统",
        "rectifier": "整流器",
        "motor-drive": "电机驱动",
        "SPWM": "正弦 PWM (SPWM)",
        "SVPWM": "空间矢量 PWM (SVPWM)",
    },
}


TASK_TYPES = [
    "definition",  # 概念与定义
    "calculation",  # 公式计算
    "design",  # 参数选型/设计
    "troubleshoot",  # 故障诊断
    "compare",  # 对比与权衡
    "explain",  # 原理讲解
]


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class SeedItem:
    id: str
    prompt: str
    language: str
    domain: str
    topic: str
    subtopic: str
    task_type: str
    difficulty: str
    created_at: str
    seed_origin: str = "synthetic-template"

    def to_json(self) -> dict:
        return asdict(self)


# -----------------------------
# Prompt templates (ZH)
# -----------------------------


TEMPLATES_ZH = {
    "definition": [
        "请用通俗但精准的语言解释{topic}中“{sub}”的定义和关键特性，并给出一个工程中的典型应用场景。",
    ],
    "calculation": [
        "已知输入电压{Vin}，输出电压{Vout}，负载电流{Iout}，开关频率{fsw}，请计算{topic}中{sub}的占空比/电感值/电容值等关键参数，并说明主要的计算步骤和假设。",
    ],
    "design": [
        "需要设计一款{topic}（子类：{sub}）电源，输入{Vin}，输出{Vout}@{Iout}，开关频率{fsw}，请给出关键器件选型思路（含安全裕量）、控制策略要点及EMI/热设计要点。",
    ],
    "troubleshoot": [
        "某{topic}（子类：{sub}）样机出现{symptom}现象，请给出可能的原因列表、定位步骤（优先级），以及可操作的改进措施。",
    ],
    "compare": [
        "在{topic}应用中，请比较“{subA}”与“{subB}”两种方案在效率、成本、动态响应、复杂度、EMI等方面的差异，并给出适用场景建议。",
    ],
    "explain": [
        "请从能量流动、关键波形与控制环路角度，系统性讲解{topic}中“{sub}”的工作原理，并指出常见误区。",
    ],
}


def random_value(symbol: str) -> str:
    # Lightweight random parameter generator for prompts
    if symbol in {"Vin", "Vout"}:
        return f"{random.choice([5, 12, 24, 48, 380])}V"
    if symbol == "Iout":
        return f"{random.choice([0.5, 2, 5, 10, 20])}A"
    if symbol == "fsw":
        return f"{random.choice([50, 100, 250, 400, 600])}kHz"
    if symbol == "symptom":
        return random.choice(["输出纹波异常增大", "环路震荡", "FET过热", "EMI超标", "效率偏低"])
    return "?"


def topic_label(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic)


def subtopic_label(topic: str, sub: str) -> str:
    return SUBTOPIC_LABELS.get(topic, {}).get(sub, sub)


def gen_prompt_zh(topic: str, sub: str, task_type: str) -> str:
    tpl = random.choice(TEMPLATES_ZH[task_type])
    # Fill placeholders if present
    readable_topic = topic_label(topic)
    readable_sub = subtopic_label(topic, sub)
    other_subtopics = [s for s in TOPICS.get(topic, []) if s != sub]
    if other_subtopics:
        sub_b = random.choice(other_subtopics)
    else:
        sub_b = sub
    readable_sub_b = subtopic_label(topic, sub_b)
    filled = tpl.format(
        topic=readable_topic,
        sub=readable_sub,
        Vin=random_value("Vin"),
        Vout=random_value("Vout"),
        Iout=random_value("Iout"),
        fsw=random_value("fsw"),
        symptom=random_value("symptom"),
        subA=readable_sub,
        subB=readable_sub_b,
    )
    return filled


def difficulty_from_task(task_type: str) -> str:
    return {
        "definition": "easy",
        "explain": "medium",
        "compare": "medium",
        "calculation": "medium",
        "design": "hard",
        "troubleshoot": "hard",
    }.get(task_type, "medium")


def make_seed_items(num: int, language: str = "zh") -> List[SeedItem]:
    random.seed(42)
    topic_keys = list(TOPICS.keys())
    created_at = datetime.now(UTC).isoformat()
    items: List[SeedItem] = []
    seen_ids = set()
    max_attempts = num * 50 if num > 0 else 0
    attempts = 0
    while len(items) < num:
        if max_attempts and attempts >= max_attempts:
            raise RuntimeError(f"Unable to generate {num} unique prompts; produced {len(items)}")
        attempts += 1
        topic = random.choice(topic_keys)
        sub = random.choice(TOPICS[topic])
        task = random.choice(TASK_TYPES)
        prompt = gen_prompt_zh(topic, sub, task)
        item_id = sha1(f"{prompt}")
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(
            SeedItem(
                id=item_id,
                prompt=prompt,
                language=language,
                domain="power-electronics",
                topic=topic,
                subtopic=sub,
                task_type=task,
                difficulty=difficulty_from_task(task),
                created_at=created_at,
            )
        )
    return items


# -----------------------------
# Teacher join + quality filters
# -----------------------------


def join_teacher_outputs(seeds_path: Path, teacher_path: Path, out_path: Path) -> None:
    seeds = {row["id"]: row for row in read_jsonl(seeds_path)}
    merged: List[dict] = []
    for row in read_jsonl(teacher_path):
        rid = row.get("id")
        resp = row.get("response") or row.get("output") or row.get("text")
        if not rid or rid not in seeds or not isinstance(resp, str):
            continue
        base = seeds[rid]
        merged.append(
            {
                "id": rid,
                "prompt": base["prompt"],
                "response": resp.strip(),
                "meta": {
                    "language": base.get("language", "zh"),
                    "domain": base.get("domain"),
                    "topic": base.get("topic"),
                    "subtopic": base.get("subtopic"),
                    "task_type": base.get("task_type"),
                    "difficulty": base.get("difficulty"),
                    "source": "teacher",
                },
            }
        )
    write_jsonl(out_path, merged)


def is_chinese_text(s: str) -> bool:
    # Heuristic: contains CJK characters
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def has_excessive_repetition(
    s: str,
    *,
    min_sentence_len: int = 12,
    repeat_threshold: int = 2,
    repetition_ratio: float = 0.45,
) -> bool:
    normalized = re.sub(r"[\s\u3000]+", " ", s).strip()
    if len(normalized) < min_sentence_len * repeat_threshold:
        return False
    parts = re.split(r"[。！？!?；;\n]+", normalized)
    sentences = [part.strip() for part in parts if len(part.strip()) >= min_sentence_len]
    if not sentences:
        return False
    counts = Counter(sentences)
    total_len = sum(len(sent) for sent in sentences)
    repeated_len = 0
    for sent, count in counts.items():
        if count >= repeat_threshold:
            repeated_len += len(sent) * (count - 1)
    if total_len == 0:
        return False
    return repeated_len / total_len >= repetition_ratio


def quality_filter(in_path: Path, out_path: Path, min_resp_len: int = 40, max_resp_len: int = 4000) -> None:
    seen_hashes = set()
    cleaned: List[dict] = []
    for row in read_jsonl(in_path):
        prompt = (row.get("prompt") or "").strip()
        resp = (row.get("response") or "").strip()
        if not prompt or not resp:
            continue
        # Length filters
        if not (min_resp_len <= len(resp) <= max_resp_len):
            continue
        # Language heuristic: prefer Chinese prompts
        if not is_chinese_text(prompt):
            continue
        # Low-quality repetition
        if has_excessive_repetition(resp):
            continue
        # Simple duplication by content hash
        h = sha1(prompt + "\n" + resp)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        cleaned.append(row)
    write_jsonl(out_path, cleaned)


def split_dataset(in_path: Path, out_dir: Path, seed: int = 1234, ratios=(0.9, 0.05, 0.05)) -> None:
    rows = list(read_jsonl(in_path))
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)


# -----------------------------
# CLI
# -----------------------------


def cmd_init(_: argparse.Namespace) -> None:
    ensure_dirs()
    # Create dataset card stub
    card = DATA / "dataset_card.md"
    if not card.exists():
        card.write_text(
            """# Power Electronics Distillation Dataset\n\n- Domain: Power electronics (CN)\n- License: specify before release\n- Source: synthetic prompts + teacher outputs\n- Intended use: SFT for small LLM distillation\n- Safety: remove PII, avoid hazardous instructions\n\n## Structure\n- data/seed: prompts for teacher\n- data/teacher_outputs: raw teacher responses\n- data/sft: cleaned + split pairs\n- data/eval: holdout evaluation set\n\n## Schema (SFT)\nEach JSONL line:\n{\n  \"id\": str,\n  \"prompt\": str,\n  \"response\": str,\n  \"meta\": {\n    \"language\": \"zh\",\n    \"domain\": \"power-electronics\",\n    \"topic\": str,\n    \"subtopic\": str,\n    \"task_type\": str,\n    \"difficulty\": str,\n    \"source\": \"teacher\"\n  }\n}\n""",
            encoding="utf-8",
        )
    print(f"Initialized dataset folders under: {DATA}")


def cmd_make_seed(ns: argparse.Namespace) -> None:
    ensure_dirs()
    num = ns.num
    seeds = make_seed_items(num=num, language="zh")
    out_path = SEED_DIR / "seed_prompts.jsonl"
    write_jsonl(out_path, (it.to_json() for it in seeds))
    print(f"Wrote {len(seeds)} seed prompts -> {out_path}")


def cmd_join(ns: argparse.Namespace) -> None:
    seeds_path = Path(ns.seeds)
    teacher_path = Path(ns.teacher)
    out_path = TEACHER_DIR / "joined.jsonl"
    join_teacher_outputs(seeds_path, teacher_path, out_path)
    print(f"Joined teacher outputs -> {out_path}")


def cmd_quality(ns: argparse.Namespace) -> None:
    in_path = Path(ns.input)
    out_path = SFT_DIR / "cleaned.jsonl"
    quality_filter(in_path, out_path, min_resp_len=ns.min_len, max_resp_len=ns.max_len)
    print(f"Quality-filtered dataset -> {out_path}")


def cmd_split(ns: argparse.Namespace) -> None:
    in_path = Path(ns.input)
    out_dir = SFT_DIR
    split_dataset(in_path, out_dir, seed=ns.seed)
    print(f"Split dataset -> {out_dir}/train.jsonl, val.jsonl, test.jsonl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset pipeline for power-electronics distillation")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Create folders and dataset card")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("make-seed", help="Generate seed prompts (ZH)")
    sp.add_argument("--num", type=int, default=120, help="Number of prompts to generate")
    sp.set_defaults(func=cmd_make_seed)

    sp = sub.add_parser("join", help="Join seed prompts with teacher outputs JSONL")
    sp.add_argument("--seeds", type=str, default=str(SEED_DIR / "seed_prompts.jsonl"))
    sp.add_argument("--teacher", type=str, required=True, help="Path to teacher outputs JSONL with fields {id, response}")
    sp.set_defaults(func=cmd_join)

    sp = sub.add_parser("quality", help="Apply quality filters")
    sp.add_argument("--input", type=str, default=str(TEACHER_DIR / "joined.jsonl"))
    sp.add_argument("--min-len", dest="min_len", type=int, default=40)
    sp.add_argument("--max-len", dest="max_len", type=int, default=4000)
    sp.set_defaults(func=cmd_quality)

    sp = sub.add_parser("split", help="Train/val/test split")
    sp.add_argument("--input", type=str, default=str(SFT_DIR / "cleaned.jsonl"))
    sp.add_argument("--seed", type=int, default=1234)
    sp.set_defaults(func=cmd_split)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
