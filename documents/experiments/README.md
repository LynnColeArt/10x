# Experiments

This folder holds experiment cards, baseline definitions, and run notes for the validation phase.

Experiments should answer one sharp question at a time. A failed experiment is useful if it clearly removes a candidate path or exposes a hidden cost.

## Baseline Contract

- [baseline-harness.md](baseline-harness.md): the dense autoregressive reference run that every candidate must beat or explain.
- Reference implementation: `thousand_to_one/` with run configs in `configs/`.

## First Experiment Cards

1. [Cramming-Style Dense Baseline](0001-cramming-style-dense-baseline.md)
2. [AdamW Versus Muon](0002-adamw-vs-muon.md)
3. [Mini DataComp/DoReMi Loop](0003-mini-datacomp-doremi-loop.md)

## Accounting Rule

No experiment gets `10x` credit unless it reports the axis it improved and the costs it may have shifted elsewhere: FLOPs, wall-clock, memory, data, preprocessing, evaluation quality, or hidden teacher cost.
