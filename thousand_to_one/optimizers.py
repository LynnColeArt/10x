from __future__ import annotations

import math
from typing import Iterable

import torch


def zeropower_via_newton_schulz5(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate matrix orthogonalization used by Muon."""
    if gradient.ndim != 2:
        raise ValueError(f"Muon expects 2D matrices, received ndim={gradient.ndim}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    compute_dtype = (
        torch.bfloat16
        if gradient.device.type == "cuda" and torch.cuda.is_bf16_supported()
        else gradient.dtype
    )
    matrix = gradient.to(dtype=compute_dtype)
    transposed = False
    if matrix.size(0) > matrix.size(1):
        matrix = matrix.T
        transposed = True
    matrix = matrix / (matrix.norm() + 1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix.to(dtype=gradient.dtype)


class Muon(torch.optim.Optimizer):
    """Moonshot-style Muon with AdamW fallback for non-matrix parameters."""

    def __init__(self, param_groups: Iterable[dict]) -> None:
        defaults = dict(
            lr=1e-3,
            weight_decay=0.1,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            muon_scale_coefficient=0.2,
            betas=(0.9, 0.95),
            eps=1e-8,
            use_muon=False,
        )
        super().__init__(param_groups, defaults)

    @staticmethod
    def adjusted_lr(
        lr: float,
        param_shape: torch.Size,
        *,
        scale_coefficient: float,
    ) -> float:
        rows, cols = param_shape[:2]
        return lr * scale_coefficient * math.sqrt(max(rows, cols))

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        scale_coefficient = group["muon_scale_coefficient"]

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.ndim > 2:
                gradient = gradient.view(gradient.size(0), -1)
            if gradient.ndim != 2:
                raise ValueError(
                    "Muon parameter groups must only contain 2D tensors. "
                    f"Received gradient ndim={gradient.ndim} for shape={tuple(parameter.shape)}"
                )

            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(gradient)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(gradient)
            update = gradient.add(buffer, alpha=momentum) if nesterov else buffer
            update = zeropower_via_newton_schulz5(update, steps=ns_steps)

            parameter.data.mul_(1 - lr * weight_decay)
            parameter.data.add_(
                update.to(dtype=parameter.dtype),
                alpha=-self.adjusted_lr(
                    lr,
                    parameter.shape,
                    scale_coefficient=scale_coefficient,
                ),
            )

    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue

            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["moment1"] = torch.zeros_like(gradient)
                state["moment2"] = torch.zeros_like(gradient)

            state["step"] += 1
            step = state["step"]
            moment1 = state["moment1"]
            moment2 = state["moment2"]

            moment1.lerp_(gradient, 1 - beta1)
            moment2.lerp_(gradient.square(), 1 - beta2)

            update = moment1 / (eps + moment2.sqrt())
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            scale = bias_correction1 / math.sqrt(bias_correction2)

            parameter.data.mul_(1 - lr * weight_decay)
            parameter.data.add_(update, alpha=-lr / scale)
