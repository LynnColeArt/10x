# Experiment Card: Cramming-Style Dense Baseline

## Question

Can we train a small dense autoregressive Transformer on commodity hardware with enough instrumentation to make later efficiency claims meaningful?

## Hypothesis

If the harness is sound, a dense AdamW baseline should produce repeatable loss curves, stable checkpoint/resume behavior, and measurable tokens/sec on the RTX 4070 before any efficiency intervention is introduced.

## Baseline

- Model: `smoke-15m`, then `probe-60m`, dense decoder-only causal LM.
- Dataset: fixed baseline corpus manifest with general text, educational/textbook text, code, and math/reasoning slices.
- Tokenizer: fixed tokenizer selected before first comparison run.
- Training budget: first fixed-token run, then fixed-wall-clock run.
- Hardware: RTX 4070 smoke/probe; DGX Spark only after local harness is stable.
- Metrics: train loss, validation loss, perplexity, wall-clock, tokens/sec, peak VRAM, data preprocessing time, checkpoint/resume success, and deterministic probe scores.

## Intervention

None. This experiment is the reference point.

The only permitted changes during this card are harness fixes that improve measurement reliability. Any recipe change that could affect model quality or speed should become a separate experiment card.

## Expected Gain

- FLOPs: None.
- Wall-clock: None.
- Memory: None.
- Data: None.
- Quality: Establishes the quality floor candidates must beat.

## Evaluation

- Language modeling: held-out validation loss and perplexity from the same corpus manifest.
- Reasoning: deterministic arithmetic and tiny GSM8K-style probes.
- Robustness: repeated seed run at the `smoke-15m` tier and checkpoint/resume test.
- Regression checks: repetition, format collapse, short-context recall, and basic code-format probes if code is included.

## Hardware Target

- RTX 4070 smoke test: confirm installation, data path, tokenizer, training loop, metrics, checkpointing, and eval.
- DGX Spark main run: run `main-130m` only after `probe-60m` produces stable local metrics.

## Failure Criteria

Downgrade the harness before testing any clever idea if:

- Training cannot resume from a checkpoint.
- Validation loss differs wildly across repeated smoke runs without explanation.
- Data loading or preprocessing cost is unmeasured.
- GPU utilization is too low to make wall-clock comparisons meaningful.
- The eval suite is too noisy to distinguish candidate methods.

## Notes

This is the anti-goblin-math experiment. It does not need to be glamorous. It needs to be repeatable, dull, and honest.

