from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import RunConfig, load_run_config
from .data import TokenWindowDataset, load_manifest, tokenize_split
from .model import TransformerLM
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Thousand To One baseline harness.")
    parser.add_argument("--config", required=True, help="Path to a JSON run configuration.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for training.seed in the run config.",
    )
    return parser.parse_args()


def apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.seed is not None:
        config.training.seed = args.seed
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision: {precision}")
    return mapping[precision]


def autocast_context(device: torch.device, dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def snapshot_gpu_utilization() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()
    return line[0].strip() if line else None


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int | None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not total_steps or total_steps <= 0:
        total_steps = warmup_steps + 1

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
    training = config.training
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("weight") and "norm" in name:
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=training.learning_rate,
        betas=(training.beta1, training.beta2),
    )


def evaluate_validation(
    model: TransformerLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast_context(device, dtype):
                _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(loss.detach().float().item())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(mean_loss),
    }


def generate_text(
    model: TransformerLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    if not tokens:
        tokens = [tokenizer.eos_token_id]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            trimmed = input_ids[:, -model.config.max_position_embeddings :]
            logits, _ = model(trimmed)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model.train()
    decoded = tokenizer.decode(input_ids[0].tolist())
    return decoded[len(prompt) :]


def evaluate_probes(
    model: TransformerLM,
    tokenizer: ByteTokenizer,
    *,
    device: torch.device,
    probe_path: Path | None,
) -> dict[str, Any]:
    if probe_path is None or not probe_path.exists():
        return {"probe_accuracy": None, "probes": []}
    with probe_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = []
    passed = 0
    for probe in payload["probes"]:
        completion = generate_text(
            model,
            tokenizer,
            probe["prompt"],
            device=device,
            max_new_tokens=probe.get("max_new_tokens", 24),
        )
        normalized_completion = completion.strip()
        expected = probe["expected"].strip()
        match_type = probe.get("match", "exact")
        if match_type == "contains":
            ok = expected in normalized_completion
        else:
            ok = normalized_completion == expected
        results.append(
            {
                "name": probe["name"],
                "prompt": probe["prompt"],
                "expected": expected,
                "completion": normalized_completion,
                "match": match_type,
                "passed": ok,
            }
        )
        passed += int(ok)
    accuracy = passed / len(results) if results else None
    return {"probe_accuracy": accuracy, "probes": results}


def checkpoint_payload(
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    tokens_seen: int,
    config: RunConfig,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config.to_dict(),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_run_note(
    path: Path,
    *,
    config: RunConfig,
    parameter_count: int,
    device: torch.device,
    dtype: torch.dtype,
    summary: dict[str, Any],
    preprocessing: dict[str, Any],
) -> None:
    lines = [
        f"# Run Note: {config.name}",
        "",
        f"- Date: {summary['started_at']}",
        f"- Run ID: {summary['run_id']}",
        f"- Seed: {summary['seed']}",
        f"- Device: {device}",
        f"- Precision: {dtype}",
        f"- Parameters: {parameter_count}",
        f"- Steps completed: {summary['steps_completed']}",
        f"- Tokens seen: {summary['tokens_seen']}",
        f"- Wall-clock seconds: {summary['train_wall_clock_seconds']:.3f}",
        f"- Peak VRAM bytes: {summary['peak_vram_bytes']}",
        f"- Final train loss: {summary['final_train_loss']}",
        f"- Final validation loss: {summary['final_validation_loss']}",
        f"- Final validation perplexity: {summary['final_validation_perplexity']}",
        f"- Probe accuracy: {summary['probe_accuracy']}",
        "",
        "## Preprocessing Ledger",
        "",
        f"- Manifest: {config.paths.manifest_path}",
        f"- Train tokenization seconds: {preprocessing['train']['elapsed_seconds']:.3f}",
        f"- Validation tokenization seconds: {preprocessing['validation']['elapsed_seconds']:.3f}",
        f"- Train tokens: {preprocessing['train']['total_tokens']}",
        f"- Validation tokens: {preprocessing['validation']['total_tokens']}",
        "",
        "## Hidden-Cost Ledger",
        "",
        "- Teacher or synthetic data cost: none in this baseline harness run.",
        "- External preprocessing cost: included only for local manifest tokenization and file reads.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_output_dir(config: RunConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{config.name}"
    output_root = Path(config.paths.output_dir)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def maybe_resume(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    checkpoint_path: Path | None,
    *,
    device: torch.device,
) -> tuple[int, int]:
    if checkpoint_path is None:
        return 0, 0
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload["step"]), int(payload["tokens_seen"])


def train(config: RunConfig) -> Path:
    set_seed(config.training.seed)
    tokenizer = ByteTokenizer()
    manifest = load_manifest(config.paths.manifest_path)
    train_split = tokenize_split(manifest["splits"]["train"], tokenizer)
    validation_split = tokenize_split(manifest["splits"]["validation"], tokenizer)
    preprocessing = {
        "train": train_split.stats,
        "validation": validation_split.stats,
        "manifest_name": manifest["name"],
    }

    device = resolve_device(config.training.device)
    dtype = resolve_dtype(config.training.precision, device)
    output_dir = build_output_dir(config)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config.to_dict())
    save_json(output_dir / "preprocessing.json", preprocessing)

    model = TransformerLM(config.model).to(device)
    if config.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
    )
    resume_from = Path(config.paths.resume_from) if config.paths.resume_from else None
    starting_step, tokens_seen = maybe_resume(
        model,
        optimizer,
        scheduler,
        resume_from,
        device=device,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and dtype == torch.float16),
    )
    train_dataset = TokenWindowDataset(train_split.tokens, config.training.sequence_length)
    validation_dataset = TokenWindowDataset(validation_split.tokens, config.training.sequence_length)
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.training.dataloader_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model.train()
    metric_log_path = output_dir / "metrics.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    train_started = time.perf_counter()
    final_train_loss = None
    validation_metrics = {"validation_loss": None, "validation_perplexity": None}
    probe_metrics = {"probe_accuracy": None, "probes": []}
    stop_reason = "max_steps"

    step = starting_step
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    while True:
        if config.training.max_steps is not None and step >= config.training.max_steps:
            stop_reason = "max_steps"
            break
        if config.training.max_tokens is not None and tokens_seen >= config.training.max_tokens:
            stop_reason = "max_tokens"
            break
        elapsed = time.perf_counter() - train_started
        if config.training.max_duration_seconds is not None and elapsed >= config.training.max_duration_seconds:
            stop_reason = "max_duration_seconds"
            break

        step += 1
        step_loss = 0.0
        step_tokens = 0
        step_started = time.perf_counter()
        for _ in range(config.training.gradient_accumulation_steps):
            try:
                inputs, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                inputs, targets = next(train_iterator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            step_tokens += int(inputs.numel())
            with autocast_context(device, dtype):
                _, loss = model(inputs, targets)
                assert loss is not None
                loss = loss / config.training.gradient_accumulation_steps
            step_loss += loss.detach().float().item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += step_tokens
        final_train_loss = step_loss
        wall_now = time.perf_counter()
        step_elapsed = wall_now - step_started
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_vram = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_vram = 0

        log_payload = {
            "step": step,
            "train_loss": step_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "tokens_seen": tokens_seen,
            "step_tokens": step_tokens,
            "step_wall_clock_seconds": step_elapsed,
            "tokens_per_second": step_tokens / max(step_elapsed, 1e-9),
            "grad_norm": float(grad_norm),
            "peak_vram_bytes": peak_vram,
        }
        with metric_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload) + "\n")

        if step % config.training.eval_every_steps == 0:
            validation_metrics = evaluate_validation(
                model,
                validation_loader,
                device=device,
                dtype=dtype,
                max_batches=config.training.max_eval_batches,
            )
            probe_path = Path(config.paths.eval_probes_path) if config.paths.eval_probes_path else None
            probe_metrics = evaluate_probes(
                model,
                tokenizer,
                device=device,
                probe_path=probe_path,
            )
            with metric_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            **validation_metrics,
                            "probe_accuracy": probe_metrics["probe_accuracy"],
                        }
                    )
                    + "\n"
                )

        if step % config.training.save_every_steps == 0:
            checkpoint = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                tokens_seen=tokens_seen,
                config=config,
            )
            torch.save(checkpoint, checkpoints_dir / f"step-{step:06d}.pt")
            torch.save(checkpoint, checkpoints_dir / "latest.pt")

    if validation_metrics["validation_loss"] is None:
        validation_metrics = evaluate_validation(
            model,
            validation_loader,
            device=device,
            dtype=dtype,
            max_batches=config.training.max_eval_batches,
        )
    if probe_metrics["probe_accuracy"] is None:
        probe_path = Path(config.paths.eval_probes_path) if config.paths.eval_probes_path else None
        probe_metrics = evaluate_probes(
            model,
            tokenizer,
            device=device,
            probe_path=probe_path,
        )

    final_checkpoint = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        tokens_seen=tokens_seen,
        config=config,
    )
    torch.save(final_checkpoint, checkpoints_dir / "latest.pt")

    summary = {
        "run_id": output_dir.name,
        "started_at": started_at,
        "seed": config.training.seed,
        "steps_completed": step,
        "tokens_seen": tokens_seen,
        "train_wall_clock_seconds": time.perf_counter() - train_started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "final_train_loss": final_train_loss,
        "final_validation_loss": validation_metrics["validation_loss"],
        "final_validation_perplexity": validation_metrics["validation_perplexity"],
        "probe_accuracy": probe_metrics["probe_accuracy"],
        "probe_results": probe_metrics["probes"],
        "gpu_utilization_pct": snapshot_gpu_utilization(),
        "stop_reason": stop_reason,
    }
    save_json(output_dir / "summary.json", summary)
    write_run_note(
        output_dir / "run-note.md",
        config=config,
        parameter_count=model.parameter_count(),
        device=device,
        dtype=dtype,
        summary=summary,
        preprocessing=preprocessing,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_run_config(args.config), args)
    output_dir = train(config)
    print(output_dir)


if __name__ == "__main__":
    main()
