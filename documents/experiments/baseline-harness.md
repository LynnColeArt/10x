# Baseline Harness

This document defines the boring baseline we need before any candidate method gets to claim an efficiency win.

The baseline is intentionally plain: a dense decoder-only Transformer trained as a causal language model with AdamW. It should be strong enough to be fair, small enough to run locally, and instrumented enough that we can tell whether a method saved compute or merely moved the cost into memory, preprocessing, data loading, or evaluation.

## Purpose

The baseline harness exists to answer three questions:

1. How much quality do we get from a conventional dense recipe under the same hardware budget?
2. What are the actual wall-clock, memory, token, and data-pipeline costs?
3. Which candidate methods improve the same capability at lower measured cost?

## Non-Negotiables

- Fixed seeds must be recorded for training, data order, initialization, and evaluation.
- Every run must record total tokens seen, optimizer steps, wall-clock time, peak VRAM, tokens/sec, final train loss, final validation loss, and evaluation scores.
- Preprocessing time, disk footprint, and filtering/tokenization cost must be logged separately from GPU training time.
- A method may claim memory accessibility without claiming training efficiency.
- A method may claim long-context inference efficiency without claiming cheaper pretraining.
- Any teacher-model, synthetic-data, or verifier cost must be counted in a hidden-cost ledger.
- The first implementation should prefer clarity and repeatability over peak performance.

## Model Family

Use a modern dense decoder-only Transformer as the reference recipe.

| Tier | Purpose | Target Scale | Hardware | Notes |
| --- | --- | --- | --- | --- |
| `smoke-15m` | Fast codepath and metric sanity | about `10M-20M` parameters | RTX 4070 | Should finish quickly enough for iteration and CI-like checks |
| `probe-60m` | First real ablations | about `50M-80M` parameters | RTX 4070 | Default tier for Muon and data-mixture probes |
| `main-130m` | DGX Spark validation | about `100M-150M` parameters | DGX Spark | Used only after smoke/probe runs are stable |

Architecture defaults:

- Causal decoder-only Transformer.
- Dense MLP and dense attention, no MoE, no recurrence, no pruning.
- Pre-norm blocks.
- RoPE positional embeddings.
- RMSNorm or LayerNorm, selected once and held fixed.
- SwiGLU or GELU MLP, selected once and held fixed.
- FlashAttention may be used if available, but must be reported.
- Context length starts at `1024` or `2048` for the first baseline; long-context curriculum is a later intervention, not baseline magic.

## Optimizer And Precision

Baseline optimizer:

- AdamW.
- Cosine or linear decay schedule, selected once and held fixed.
- Warmup steps recorded.
- Gradient clipping setting recorded.

Baseline precision:

- Use the most stable mixed precision supported by the hardware stack.
- Record BF16, FP16, TF32, FP32, and any compiler settings.
- Do not count low-precision speedups as a candidate intervention unless the baseline and intervention differ intentionally.

## Dataset Contract

The baseline should use a fixed corpus manifest rather than an implicit dataset download.

Initial corpus target:

- A small general-language slice.
- A small high-quality educational or textbook-like slice.
- A small code slice.
- A small math/reasoning slice.
- A held-out validation split from the same manifest.

Rules:

- No synthetic teacher data in the baseline.
- No adaptive filtering in the baseline.
- Tokenization must be fixed before comparing candidate methods.
- Dataset order must be reproducible.
- Deduplication, filtering, tokenization, and packing time must be measured.
- Data-mixture experiments may change sampling weights, but they must keep source manifests and preprocessing accounting visible.

## Budgets

Each candidate should be evaluated under at least two budget views.

| Budget View | Meaning | Why It Matters |
| --- | --- | --- |
| Fixed tokens | Same number of training tokens | Shows quality-per-token and optimizer/data effects |
| Fixed wall-clock | Same elapsed training time | Shows practical user-facing efficiency |
| Fixed target loss | Time/tokens needed to reach a baseline loss | Shows convergence speed |
| Fixed memory | Largest stable batch/model under a VRAM cap | Separates accessibility from speed |

The first harness implementation should support fixed-token and fixed-wall-clock runs before anything more exotic.

## Metrics

Required training metrics:

- Total elapsed wall-clock time.
- Data preprocessing time.
- Dataset loading and tokenization time.
- Training tokens/sec.
- Optimizer steps/sec.
- GPU utilization when available.
- Peak allocated VRAM.
- Final training loss.
- Final validation loss and perplexity.
- Checkpoint size.
- Resume success or failure.

Required evaluation metrics:

- Held-out language-model validation loss.
- Small deterministic arithmetic probes.
- Small deterministic instruction-following probes.
- Small code or structured-output probes if the training corpus includes code.
- Regression prompts that check repetition, format collapse, and short-context recall.

Optional evaluation metrics:

- Tiny GSM8K-style exact-match subset.
- Tiny HumanEval-style syntax-only or unit-test subset.
- Long-context retrieval probe once context curriculum becomes part of the experiment.

## Run Record Template

Each completed run should create a run note with:

- Run ID:
- Date:
- Git commit:
- Hardware:
- Driver/CUDA/PyTorch stack:
- Model tier:
- Parameter count:
- Dataset manifest:
- Tokenizer:
- Training budget:
- Precision:
- Optimizer:
- Peak VRAM:
- Tokens/sec:
- Wall-clock:
- Final train loss:
- Final validation loss:
- Eval summary:
- Preprocessing ledger:
- Hidden-cost ledger:
- Failure notes:

## First Implementation Order

1. Implement or select the minimal dense Transformer training harness.
2. Make `smoke-15m` run end-to-end with metrics and a saved run note.
3. Scale to `probe-60m` on RTX 4070.
4. Run the AdamW baseline twice with different seeds to estimate variance.
5. Only then compare Muon, data-mixture weighting, sparse training, recurrence, or curriculum.

## Failure Criteria

Pause candidate experiments if the baseline has any of these failures:

- The run cannot resume from checkpoint.
- Validation loss is not reproducible within a reasonable seed variance.
- Data loading dominates wall-clock time without being measured.
- GPU utilization is consistently low and unexplained.
- Evaluation prompts are unstable or manually judged.
- A candidate method changes more than one variable at once.

