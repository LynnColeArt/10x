# Run Note: smoke-15m-real-v1 variance

## Status

The first variance pass for the real-source `smoke-15m` baseline is complete.

We now have two runs of the same config on the RTX 4070 with different seeds and broadly similar outcomes.

## Runs Compared

### Run A

- Run ID: `20260502T142515Z-smoke-15m-real-v1`
- Seed: `1234`
- Raw artifacts: `runs/20260502T142515Z-smoke-15m-real-v1`

### Run B

- Run ID: `20260503T083545Z-smoke-15m-real-v1`
- Seed: `4321`
- Raw artifacts: `runs/20260503T083545Z-smoke-15m-real-v1`

## Shared Setup

- Config: `configs/smoke-15m-real-v1.json`
- GPU: NVIDIA GeForce RTX 4070 `12 GB`
- Precision: `bfloat16`
- Parameters: `16,350,768`
- Sequence length: `512`
- Micro-batch size: `4`
- Steps: `60`
- Tokens seen: `122,880`
- Corpus manifest: `data/real-smoke-v1/manifest.json`

## Outcome Comparison

| Metric | Seed `1234` | Seed `4321` | Delta |
| --- | --- | --- | --- |
| Wall-clock seconds | `3.816` | `3.984` | `+0.168` |
| Overall tokens/sec | `32,204` | `30,843` | `-1,361` |
| Peak VRAM bytes | `705,562,624` | `705,562,624` | `0` |
| First train loss | `5.5931` | `5.6656` | `+0.0725` |
| Final train loss | `3.1522` | `3.2740` | `+0.1217` |
| Final validation loss | `3.3011` | `3.2664` | `-0.0346` |
| Final validation perplexity | `27.1416` | `26.2176` | `-0.9240` |
| Probe accuracy | `0.0` | `0.0` | `0.0` |

## Interpretation

This is a healthy first variance result.

- Memory behavior was identical across both runs.
- Wall-clock moved by about `4.4%`, which is noticeable but not alarming for a short smoke run with eval and checkpoint overhead.
- Final validation loss differed by about `0.035`, which is small enough that the harness looks stable at this scale.
- Final train loss moved more than validation loss, which suggests short-run minibatch order and seed effects are real but not dominating the overall outcome.

The two runs do **not** prove the baseline is fully characterized yet, but they are enough to justify moving to the next comparison stage. We are no longer looking at a single lucky run.

## What This Proves

- The real-source smoke baseline is reproducible enough to use as a reference for the next experiment.
- The current harness does not show obvious seed-driven instability under this short budget.
- We can now compare another optimizer against the dense AdamW baseline without feeling like we are standing on one anecdote.

## Remaining Limits

- These are still short `60`-step smoke runs, not serious capability runs.
- Probe accuracy remains `0.0`, so this pass says nothing about reasoning preservation yet.
- The baseline contract still wants stricter sequence-length and longer-run decisions before we call it a mature benchmark.

## Next Step

The next slice should be `AdamW vs Muon` on top of this stabilized smoke baseline:

- keep the real-smoke corpus fixed,
- keep the `smoke-15m` architecture fixed,
- keep the step budget fixed,
- and measure whether Muon improves time-to-loss or validation loss enough to justify its extra optimizer complexity.

