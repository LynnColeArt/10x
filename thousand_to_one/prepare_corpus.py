from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GENERAL_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
GENERAL_VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
MATH_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
MATH_VALID_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


@dataclass(frozen=True)
class BuildSpec:
    train_limit: int
    validation_limit: int


DEFAULT_SPECS = {
    "general": BuildSpec(train_limit=2500, validation_limit=250),
    "education": BuildSpec(train_limit=1500, validation_limit=150),
    "code": BuildSpec(train_limit=1500, validation_limit=150),
    "math": BuildSpec(train_limit=1500, validation_limit=150),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local real-smoke corpus manifest.")
    parser.add_argument(
        "--output-dir",
        default="data/real-smoke-v1",
        help="Directory where the manifest and JSONL shards will be written.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic shuffling seed.")
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_wikitext(text: str) -> list[str]:
    records = []
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        if len(line) < 40:
            continue
        if line.startswith("="):
            continue
        records.append(line)
    return records


def load_education_records() -> list[str]:
    topics_module = importlib.import_module("pydoc_data.topics")
    topics_dict = getattr(topics_module, "topics")
    records: list[str] = []
    for topic_name, topic_text in topics_dict.items():
        for piece in re.split(r"\n\s*\n", topic_text):
            piece = normalize_whitespace(piece)
            if len(piece) < 120:
                continue
            if piece.count(">>>") > 2:
                continue
            records.append(f"{topic_name}: {piece}")
    return records


def chunk_code_text(text: str, *, max_chars: int = 2400, max_lines: int = 80) -> list[str]:
    chunks: list[str] = []
    current_lines: list[str] = []
    current_chars = 0
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line and not current_lines:
            continue
        projected = current_chars + len(line) + 1
        if current_lines and (projected > max_chars or len(current_lines) >= max_lines):
            chunk = "\n".join(current_lines).strip()
            if len(chunk) >= 160:
                chunks.append(chunk)
            current_lines = []
            current_chars = 0
            if not line:
                continue
        current_lines.append(line)
        current_chars += len(line) + 1
    if current_lines:
        chunk = "\n".join(current_lines).strip()
        if len(chunk) >= 160:
            chunks.append(chunk)
    return chunks


def load_code_records(seed: int) -> tuple[list[str], list[str]]:
    root = Path("/usr/lib/python3.12")
    file_paths = [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "test" not in path.parts
        and "dist-packages" not in path.parts
        and path.name != "__init__.py"
    ]
    rng = random.Random(seed)
    rng.shuffle(file_paths)
    validation_files = file_paths[:30]
    train_files = file_paths[30:280]

    def materialize(paths: Iterable[Path]) -> list[str]:
        records: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            for chunk in chunk_code_text(text):
                records.append(f"# path: {path}\n{chunk}")
        return records

    return materialize(train_files), materialize(validation_files)


def load_math_records(text: str) -> list[str]:
    records = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        payload = json.loads(raw_line)
        question = normalize_whitespace(payload["question"])
        answer = normalize_whitespace(payload["answer"])
        records.append(f"Question: {question}\nAnswer: {answer}")
    return records


def deterministic_sample(records: list[str], limit: int, seed: int) -> list[str]:
    if len(records) <= limit:
        return records
    rng = random.Random(seed)
    selected = records[:]
    rng.shuffle(selected)
    return selected[:limit]


def deterministic_partition(
    records: list[str],
    *,
    train_limit: int,
    validation_limit: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    validation_records = shuffled[:validation_limit]
    train_records = shuffled[validation_limit : validation_limit + train_limit]
    return train_records, validation_records


def write_jsonl(path: Path, records: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"text": record}, ensure_ascii=False) + "\n")


def build_manifest(output_dir: Path) -> dict[str, object]:
    return {
        "name": "real-smoke-v1",
        "description": "Real-source smoke corpus built from WikiText-2, local Python docs, local stdlib code, and GSM8K raw JSONL.",
        "splits": {
            "train": [
                {"path": "train/general.jsonl", "source": "general"},
                {"path": "train/education.jsonl", "source": "education"},
                {"path": "train/code.jsonl", "source": "code"},
                {"path": "train/math.jsonl", "source": "math"},
            ],
            "validation": [
                {"path": "validation/general.jsonl", "source": "general"},
                {"path": "validation/education.jsonl", "source": "education"},
                {"path": "validation/code.jsonl", "source": "code"},
                {"path": "validation/math.jsonl", "source": "math"},
            ],
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    general_train = split_wikitext(fetch_text(GENERAL_TRAIN_URL))
    general_validation = split_wikitext(fetch_text(GENERAL_VALID_URL))
    education_records = load_education_records()
    code_train, code_validation = load_code_records(args.seed)
    math_train = load_math_records(fetch_text(MATH_TRAIN_URL))
    math_validation = load_math_records(fetch_text(MATH_VALID_URL))

    education_train, education_validation = deterministic_partition(
        education_records,
        train_limit=DEFAULT_SPECS["education"].train_limit,
        validation_limit=DEFAULT_SPECS["education"].validation_limit,
        seed=args.seed + 2,
    )

    sources = {
        "general": (
            deterministic_sample(general_train, DEFAULT_SPECS["general"].train_limit, args.seed),
            deterministic_sample(general_validation, DEFAULT_SPECS["general"].validation_limit, args.seed + 1),
        ),
        "education": (
            education_train,
            education_validation,
        ),
        "code": (
            deterministic_sample(code_train, DEFAULT_SPECS["code"].train_limit, args.seed + 4),
            deterministic_sample(code_validation, DEFAULT_SPECS["code"].validation_limit, args.seed + 5),
        ),
        "math": (
            deterministic_sample(math_train, DEFAULT_SPECS["math"].train_limit, args.seed + 6),
            deterministic_sample(math_validation, DEFAULT_SPECS["math"].validation_limit, args.seed + 7),
        ),
    }

    for source_name, (train_records, validation_records) in sources.items():
        write_jsonl(train_dir / f"{source_name}.jsonl", train_records)
        write_jsonl(validation_dir / f"{source_name}.jsonl", validation_records)

    manifest = build_manifest(output_dir)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summary = {
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sources": {
            source_name: {
                "train_records": len(train_records),
                "validation_records": len(validation_records),
                "train_chars": sum(len(record) for record in train_records),
                "validation_chars": sum(len(record) for record in validation_records),
            }
            for source_name, (train_records, validation_records) in sources.items()
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(output_dir)


if __name__ == "__main__":
    main()
