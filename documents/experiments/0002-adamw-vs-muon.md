# Experiment Card: AdamW Versus Muon

## Question

Does Muon improve convergence or wall-clock efficiency for small local Transformer pretraining compared with a strong AdamW baseline?

## Hypothesis

If Muon is useful below frontier scale, it should reach the same validation loss as AdamW using fewer optimizer steps or less wall-clock time after accounting for Newton-Schulz overhead and any stability tuning.

## Baseline

- Model: `probe-60m` dense decoder-only causal LM from the baseline harness.
- Dataset: fixed baseline corpus manifest.
- Tokenizer: same tokenizer as baseline.
- Training budget: matched fixed tokens, matched fixed wall-clock, and target-loss comparison.
- Hardware: RTX 4070 first; DGX Spark only if local results are positive or ambiguous.
- Metrics: validation loss over time, tokens/sec, optimizer step time, peak VRAM, optimizer-state memory, gradient norms, loss spikes, and checkpoint/resume success.

## Intervention

Replace AdamW with Muon for eligible matrix parameters while retaining AdamW for parameters that should not use Muon.

Initial rule:

- Muon for dense linear weight matrices.
- AdamW for embeddings, output head, normalization parameters, scalar parameters, and biases.
- Record Newton-Schulz iteration count and precision.
- Report optimizer overhead separately from forward/backward time.

## Expected Gain

- FLOPs: Ambiguous. Muon adds optimizer computation, so raw FLOPs may increase.
- Wall-clock: Target `1.1x-1.5x` faster time-to-validation-loss before considering it promising.
- Memory: Possible optimizer-state difference, but not the primary claim.
- Data: No data reduction unless it reaches target quality in fewer tokens.
- Quality: Equal or better validation loss and no regression on deterministic probes.

## Evaluation

- Language modeling: validation loss curves at matched tokens and matched wall-clock.
- Reasoning: deterministic arithmetic and tiny reasoning probes from the baseline harness.
- Robustness: loss-spike count, gradient-norm outliers, and at least two seeds if the first result is promising.
- Regression checks: checkpoint/resume, final sample sanity, and probe stability.

## Hardware Target

- RTX 4070 smoke test: small `smoke-15m` run to verify implementation and overhead.
- RTX 4070 probe run: `probe-60m` AdamW versus Muon.
- DGX Spark main run: `main-130m` only if RTX results show credible time-to-loss improvement or better stability.

## Failure Criteria

Downgrade Muon if:

- Newton-Schulz overhead erases convergence gains.
- Muon requires fragile hyperparameter tuning to match AdamW.
- Validation loss improves only under fixed tokens but loses under fixed wall-clock.
- Memory use or checkpointing becomes unreliable.
- Probe evals regress while validation loss improves.

## Notes

This card is motivated by DeepSeek-V4, but we must not inherit DeepSeek-scale conclusions. The local question is narrower: does Muon help our hardware and model sizes enough to earn a place in the stack?

