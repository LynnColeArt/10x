# Run Note: AdamW Versus Muon on `smoke-15m`

## Status

The first `AdamW vs Muon` smoke comparison is complete.

We now have matched two-seed runs for both optimizers on the same real-source `smoke-15m` setup on the RTX 4070.

## Runs Compared

### AdamW, seed `1234`

- Run ID: `20260502T142515Z-smoke-15m-real-v1`
- Raw artifacts: `runs/20260502T142515Z-smoke-15m-real-v1`
- Note: this run predates the explicit `seed` field in `summary.json`; we treat it as seed `1234` because `configs/smoke-15m-real-v1.json` used the default seed and the earlier tracked note already recorded it that way.

### AdamW, seed `4321`

- Run ID: `20260503T083545Z-smoke-15m-real-v1`
- Raw artifacts: `runs/20260503T083545Z-smoke-15m-real-v1`

### Muon, seed `1234`

- Run ID: `20260503T094634Z-smoke-15m-real-v1-muon`
- Raw artifacts: `runs/20260503T094634Z-smoke-15m-real-v1-muon`

### Muon, seed `4321`

- Run ID: `20260503T094654Z-smoke-15m-real-v1-muon`
- Raw artifacts: `runs/20260503T094654Z-smoke-15m-real-v1-muon`

## Shared Setup

- AdamW config: `configs/smoke-15m-real-v1.json`
- Muon config: `configs/smoke-15m-real-v1-muon.json`
- GPU: NVIDIA GeForce RTX 4070 `12 GB`
- Precision: `bfloat16`
- Parameters: `16,350,768`
- Sequence length: `512`
- Micro-batch size: `4`
- Steps: `60`
- Tokens seen: `122,880`
- Corpus manifest: `data/real-smoke-v1/manifest.json`

## Outcome Comparison

| Optimizer | Seed | Wall-clock seconds | Overall tokens/sec | Peak VRAM bytes | Final train loss | Final validation loss | Final validation perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW | `1234` | `3.816` | `32,204` | `705,562,624` | `3.1522` | `3.3011` | `27.1416` |
| AdamW | `4321` | `3.984` | `30,843` | `705,562,624` | `3.2740` | `3.2664` | `26.2176` |
| Muon | `1234` | `4.480` | `27,426` | `638,932,992` | `3.1827` | `3.2631` | `26.1297` |
| Muon | `4321` | `4.433` | `27,719` | `638,932,992` | `3.2590` | `3.2327` | `25.3489` |

## Mean Comparison

| Metric | AdamW mean | Muon mean | Delta |
| --- | --- | --- | --- |
| Wall-clock seconds | `3.900` | `4.457` | `+0.557` |
| Overall tokens/sec | `31,524` | `27,572` | `-3,951` |
| Peak VRAM bytes | `705,562,624` | `638,932,992` | `-66,629,632` |
| Final train loss | `3.2131` | `3.2209` | `+0.0078` |
| Final validation loss | `3.2837` | `3.2479` | `-0.0358` |
| Final validation perplexity | `26.6796` | `25.7393` | `-0.9403` |

## Interpretation

This is a useful but mixed result.

- Muon beat AdamW on final validation loss for both seeds, by about `0.0380` for seed `1234` and `0.0337` for seed `4321`.
- Muon also reduced peak VRAM by `66,629,632` bytes, about `9.4%`, in both runs.
- Muon lost on wall-clock in both runs. The slowdown was about `0.665` seconds for seed `1234` and `0.449` seconds for seed `4321`, with an average slowdown of about `14.3%`.
- Probe accuracy stayed `0.0` everywhere, so this slice still says nothing about reasoning preservation.

The practical reading is:

- Muon is **not** a wall-clock win at the current `smoke-15m` budget.
- Muon **is** a credible quality-per-token and memory candidate at this scale.
- The quality gain is real enough across two seeds that the optimizer should stay in the queue.

## What This Proves

- The harness can cleanly swap between AdamW and Muon while preserving the rest of the baseline contract.
- Muon is not obviously unstable on our local dense baseline.
- At small scale, Muon seems to buy a modest validation-loss improvement and lower VRAM, but not faster training.

## Remaining Limits

- These are still only `60`-step smoke runs.
- The run is too short to say whether Muon reaches a target loss in fewer total steps at larger scale.
- We did not isolate optimizer-step overhead from total training time yet.
- We still have no signal on reasoning retention.

## Next Step

The next Muon slice should be the `probe-60m` comparison:

- add a Muon variant of the `probe-60m` config,
- keep the corpus and token budget fixed,
- compare validation loss at matched tokens and matched wall-clock,
- and check whether the small smoke quality gain survives once the run is long enough for time-to-loss to matter.
