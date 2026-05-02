# Configs

These JSON configs drive the local training harness in `thousand_to_one/`.

## Included Configs

- `tiny-dev-byte.json`: tiny verification run used to prove the harness works end-to-end.
- `smoke-15m-byte.json`: smoke-tier scaffold near the `15M` parameter target.
- `probe-60m-byte.json`: probe-tier scaffold near the `60M` parameter target.
- `main-130m-byte.json`: DGX Spark-oriented scaffold near the `130M` parameter target.

## Important Caveat

The included configs point at the small sample manifest in `fixtures/sample-baseline/` and use a short sequence length so the harness can be exercised locally without external data dependencies.

They are verification-scale defaults, not the final research baseline.

Before a serious baseline run, we should swap in:

- a real corpus manifest,
- the agreed tokenizer choice,
- a sequence length that matches the baseline contract,
- and hardware-appropriate batch and duration budgets.

