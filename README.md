# Thousand To One

Research and experimental planning for finding practical paths to `10x+` LLM training efficiency on commodity-accessible hardware.

The project starts from the hypothesis that modern LLM training recipes are not globally optimal for smaller labs, independent researchers, or high-end local machines. The goal is to synthesize scattered research from 2022-2026 into testable training recipes, then validate the most promising candidates experimentally.

Primary documentation lives in [documents/](documents/README.md).

## Baseline Harness

The repo now includes a minimal dense causal-language-model harness under `thousand_to_one/`.

Quick verification run:

```bash
python3 -m thousand_to_one.train --config configs/tiny-dev-byte.json
```

Prepare the first real smoke corpus:

```bash
python3 -m thousand_to_one.prepare_corpus --output-dir data/real-smoke-v1
```

Run the first real-source smoke baseline:

```bash
python3 -m thousand_to_one.train --config configs/smoke-15m-real-v1.json
```

Run the same config with a different seed:

```bash
python3 -m thousand_to_one.train --config configs/smoke-15m-real-v1.json --seed 4321
```

Larger baseline configs:

- `configs/smoke-15m-byte.json`
- `configs/probe-60m-byte.json`
- `configs/main-130m-byte.json`
