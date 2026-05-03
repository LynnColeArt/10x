from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    manifest_path: str
    eval_probes_path: str | None = None
    output_dir: str = "runs"
    resume_from: str | None = None


@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_position_embeddings: int = 2048
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    rope_theta: float = 10000.0
    vocab_size: int = 260


@dataclass
class TrainingConfig:
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int | None = None
    max_tokens: int | None = None
    max_duration_seconds: float | None = None
    optimizer: str = "adamw"
    log_every_steps: int = 1
    eval_every_steps: int = 10
    save_every_steps: int = 10
    max_eval_batches: int | None = None
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_scale_coefficient: float = 0.2
    device: str = "auto"
    precision: str = "auto"
    compile_model: bool = False
    seed: int = 1234
    dataloader_workers: int = 0


@dataclass
class RunConfig:
    name: str
    description: str
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    payload = _load_json(config_path)
    return RunConfig(
        name=payload["name"],
        description=payload.get("description", ""),
        paths=PathsConfig(**payload["paths"]),
        model=ModelConfig(**payload["model"]),
        training=TrainingConfig(**payload["training"]),
        tags=payload.get("tags", []),
    )
