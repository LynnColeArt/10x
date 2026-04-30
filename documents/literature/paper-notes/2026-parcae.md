# Paper Note: Parcae

## Citation

- Title: Parcae: Scaling Laws For Stable Looped Language Models
- Authors: Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu
- Year: 2026
- Link: https://arxiv.org/abs/2604.12946
- Code: https://github.com/sandyresearch/parcae

## One-Sentence Claim

A stabilized looped language-model architecture can reuse parameters across repeated computation steps, improving quality under fixed parameter/data budgets and enabling predictable training and test-time scaling laws for loops.

## Efficiency Axis

- FLOPs: Does not simply reduce FLOPs; it introduces looping as a separate compute axis while keeping parameter count fixed.
- Wall-clock: Training and inference may become slower at a fixed parameter count because loops add computation.
- Memory: Potentially strong parameter-memory efficiency because depth is reused rather than stored as unique layers.
- Energy/cost: Ambiguous; may save memory/parameter cost but spend extra compute through loops.
- Data required: Paper suggests looping and data should be scaled together under fixed FLOP budgets.
- Hidden teacher or preprocessing cost: None obvious from the architecture claim, but full scaling-law reproduction is expensive.

## Training Regime

- Pretraining: Yes, looped language-model training.
- Continued pretraining: Not primary.
- Fine-tuning: Not primary.
- Pruning: No.
- Inference-only: No; includes training and test-time looping.
- Evaluation-only: No.

## Baseline

The paper compares against prior looped models and strong Transformer baselines under fixed parameter and data budgets. This is directly relevant to our recurrent-depth lane, though the work is very new and needs reproduction.

## Evidence

The paper frames traditional fixed-depth scaling as increasing quality through more parameters, data, or training FLOPs. It identifies instability in previous looped architectures, especially residual explosion and loss spikes, and proposes spectral-norm-constrained injection parameters via a negative diagonal parameterization.

Reported headline results include up to `6.3%` lower validation perplexity versus prior large-scale looped models. At 1.3B parameters, the abstract reports CORE and Core-Extended gains of `2.99` and `1.18` points over strong Transformer baselines under fixed parameter and data budgets, reaching up to `87.5%` of the quality of a Transformer twice the size.

The public repository provides installable models and training configs, including `140M`, `370M`, `770M`, and `1.3B` model references.

## Capability Preservation

Capability preservation is measured through language-model loss and CORE/Core-Extended style benchmark quality. This is relevant but not yet a complete reasoning evaluation.

## Hardware Reality

Parcae is promising for memory-constrained hardware because repeated depth can reuse parameters. The danger is compute: a looped model can be cheaper in memory but more expensive in wall-clock per token. For commodity training, the key question is whether parameter reuse lets us train a model that would otherwise not fit, and whether the resulting quality-per-dollar beats a smaller dense model.

## Composability

Likely composes with:

- Data curation.
- Optimizer memory reduction.
- Low precision.
- Adaptive loop count.
- Possibly structured sparse MLPs inside recurrent blocks.

Likely conflicts with:

- Wall-clock-limited training budgets.
- Sparse training methods that destabilize recurrent dynamics.
- Any `10x` accounting that counts parameter reduction but ignores extra loop FLOPs.

## Reproduction Path

Use the public repository for a tiny smoke test if dependency friction is manageable. Our first research experiment should not try to reproduce the full scaling law. Instead:

- Compare a tiny dense Transformer and a tiny looped model at matched parameter, token, and wall-clock budgets.
- Track loss, memory, tokens/sec, and reasoning probes.
- Sweep loop count during training and inference separately.
- Check whether extra test-time loops improve hard tasks enough to justify inference cost.

## Verdict

- Evidence strength: Medium-high as a fresh LLM architecture signal; reproduction still needed.
- Estimated usable gain: Potentially `1.5x-2x` parameter efficiency, not automatically training efficiency. It may become part of a `10x` stack if paired with data efficiency and memory-aware optimizers.
- Risk: It may move cost from parameters into sequential compute, which could hurt commodity wall-clock.
- Follow-up: Treat as the top recurrent-depth candidate, but score it separately on parameter efficiency, training compute, and inference compute.
