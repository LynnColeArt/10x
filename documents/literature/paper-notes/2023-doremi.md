# Paper Note: DoReMi

## Citation

- Title: DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- Authors: Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu
- Year: 2023, NeurIPS
- Link: https://arxiv.org/abs/2305.10429
- Code: https://github.com/sangmichaelxie/doremi

## One-Sentence Claim

A small proxy model trained with domain-robust optimization can choose data mixture weights that make a much larger language model train faster and perform better.

## Efficiency Axis

- FLOPs: Reports reaching baseline downstream accuracy with `2.6x` fewer training steps on The Pile.
- Wall-clock: Fewer steps should reduce wall-clock, but proxy training and data resampling cost must be counted.
- Memory: No direct memory reduction.
- Energy/cost: Potential training-cost reduction from fewer steps, plus extra proxy-model cost.
- Data required: Same broad data sources, but resampled by learned domain weights.
- Hidden teacher or preprocessing cost: Trains a proxy model and uses a reference model to estimate excess loss.

## Training Regime

- Pretraining: Yes.
- Continued pretraining: Not primary.
- Fine-tuning: No.
- Pruning: No.
- Inference-only: No.
- Evaluation-only: No.

## Baseline

The baseline is a larger model trained on default domain weights, especially The Pile's default mixture. This is a relevant LLM pretraining baseline, though the method assumes domains are available and meaningful.

## Evidence

The paper uses a `280M` proxy model to set domain weights for an `8B` model, about `30x` larger. The arXiv abstract reports improved perplexity across all Pile domains, a `6.5` percentage-point improvement in average few-shot downstream accuracy over The Pile's default weights, and baseline accuracy with `2.6x` fewer training steps. On the GLaM dataset, the method matches downstream-task-tuned domain weights without using downstream-task labels.

The public implementation describes DoReMi as a black-box tool that outputs optimized domain weights and includes domain-weighted sampling, downstream evaluation, and Hugging Face Trainer plus FlashAttention2 integration.

## Capability Preservation

Capability is measured by perplexity across domains and few-shot downstream accuracy. This is directly relevant to language-model training, though not yet sufficient for product-level reasoning preservation.

## Accidental Or Side-Effect Signal

- Original goal: Optimize data mixture proportions without knowing downstream tasks.
- Unexpected or secondary finding: A relatively small proxy can improve training of a model `30x` larger.
- Why it matters for `10x+` training efficiency: It suggests cheap proxy training can steer expensive full training, which is a general pattern we may reuse for data, masks, architectures, and curricula.
- Risk of overinterpreting the side effect: Proxy-model weights may not transfer cleanly to smaller commodity settings, and reference/proxy costs can eat the gain.

## Hardware Reality

The method is attractive for commodity training because the expensive part is data selection rather than exotic kernels. However, it depends on domain-labeled datasets, enough compute to train a proxy, and data pipelines that can resample efficiently.

## Composability

Likely composes with:

- DataComp-LM-style curation.
- Sparse training.
- Recurrent-depth models.
- Cramming-style local baselines.
- Small proxy experiments on RTX 4070.

Likely conflicts with:

- Undifferentiated datasets without reliable domain labels.
- Highly synthetic curricula where "domain" is not a stable concept.
- Accounting that ignores the reference/proxy training cost.

## Reproduction Path

Run a miniature DoReMi-style experiment:

- Split a small corpus into coarse domains.
- Train a cheap proxy model with domain reweighting.
- Train a tiny target model on default versus optimized mixtures.
- Compare tokens-to-loss, downstream probes, wall-clock, and proxy overhead.

This is one of the best early RTX 4070 smoke-test candidates.

## Verdict

- Evidence strength: High for data-mixture impact on LLM pretraining.
- Estimated usable gain: `1.5x-2.6x` training-step efficiency is plausible in a controlled local setting; larger claims require careful accounting.
- Risk: Domain labels and reference/proxy costs may make the method less simple than the headline suggests.
- Follow-up: Promote to first-wave experiment-card candidate after DataComp-LM.
