from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


@dataclass
class ManifestEntry:
    path: Path
    source: str


@dataclass
class SplitLoadResult:
    tokens: torch.Tensor
    stats: dict[str, Any]


class TokenWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokens: torch.Tensor, sequence_length: int) -> None:
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.window_count = max(0, len(tokens) - sequence_length - 1)
        if self.window_count <= 0:
            raise ValueError(
                "Not enough tokens for the requested sequence length. "
                f"Need more than {sequence_length + 1}, found {len(tokens)}."
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        stop = start + self.sequence_length + 1
        window = self.tokens[start:stop]
        inputs = window[:-1].to(torch.long)
        targets = window[1:].to(torch.long)
        return inputs, targets


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base_dir = manifest_path.parent
    splits: dict[str, list[ManifestEntry]] = {}
    for split_name, entries in payload["splits"].items():
        split_entries: list[ManifestEntry] = []
        for entry in entries:
            entry_path = (base_dir / entry["path"]).resolve()
            source = entry.get("source", Path(entry["path"]).stem)
            split_entries.append(ManifestEntry(path=entry_path, source=source))
        splits[split_name] = split_entries
    return {
        "name": payload["name"],
        "description": payload.get("description", ""),
        "splits": splits,
    }


def _iter_texts(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        texts: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                texts.append(payload["text"])
        return texts
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def tokenize_split(
    entries: list[ManifestEntry],
    tokenizer: ByteTokenizer,
) -> SplitLoadResult:
    started_at = time.perf_counter()
    encoded: list[int] = []
    file_stats: list[dict[str, Any]] = []
    total_texts = 0
    for entry in entries:
        texts = _iter_texts(entry.path)
        total_texts += len(texts)
        token_count_before = len(encoded)
        for text in texts:
            encoded.extend(tokenizer.encode(text, add_bos=False, add_eos=True))
        file_stats.append(
            {
                "path": str(entry.path),
                "source": entry.source,
                "texts": len(texts),
                "tokens": len(encoded) - token_count_before,
            }
        )
    elapsed = time.perf_counter() - started_at
    tokens = torch.tensor(encoded, dtype=torch.int32)
    return SplitLoadResult(
        tokens=tokens,
        stats={
            "elapsed_seconds": elapsed,
            "total_texts": total_texts,
            "total_tokens": int(tokens.numel()),
            "files": file_stats,
        },
    )

