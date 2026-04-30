# Paper Note: Cramming

## Citation

- Title: Cramming: Training a Language Model on a Single GPU in One Day
- Authors: Jonas Geiping, Tom Goldstein
- Year: 2022
- Link: https://arxiv.org/abs/2212.14034
- Code: https://github.com/JonasGeiping/cramming

## One-Sentence Claim

Re-optimizing the full pretraining pipeline for a fixed single-GPU, 24-hour budget can produce a BERT-style language model from scratch with surprisingly competitive downstream performance.

## Efficiency Axis

- FLOPs: Fixed by a one-GPU, one-day training budget.
- Wall-clock: Directly optimized around 24 hours.
- Memory: Single consumer GPU constraint forces practical memory discipline.
- Energy/cost: Directly relevant to commodity hardware.
- Data required: Uses raw text with preprocessing and tokenization outside the training budget under its rules.
- Hidden teacher or preprocessing cost: Preprocessing, tokenizer construction, filtering, and downstream fine-tuning are excluded from the 24-hour training budget.

## Training Regime

- Pretraining: Yes, masked language model pretraining from scratch.
- Continued pretraining: No.
- Fine-tuning: Downstream GLUE fine-tuning is used for evaluation, but excluded from the main compute budget.
- Pruning: No.
- Inference-only: No.
- Evaluation-only: No.

## Baseline

The baseline is not a frontier LLM recipe; it is a constrained-compute BERT-style training setup. This makes it unusually relevant to our commodity-hardware target even though it is masked LM rather than autoregressive reasoning-model training.

## Evidence

The arXiv abstract says the paper re-analyzes nearly all components of the pretraining pipeline for the single-GPU scenario, provides a modified pipeline with performance close to BERT, and studies which modern improvements actually help in limited compute. The repository encodes explicit rules: no pretrained models in the pipeline, single GPU for 24 hours, GLUE evaluation, and careful separation between training budget and preprocessing.

The repository also includes an important working note: data processing and filtering may be under-explored compared with training and architecture tweaks.

## Capability Preservation

Capability is measured through GLUE-style downstream evaluation after brief fine-tuning. This is useful for baseline discipline, but it is not an autoregressive reasoning or instruction-following evaluation.

## Accidental Or Side-Effect Signal

- Original goal: Ask how far a language model can get under a strict single-GPU training budget.
- Unexpected or secondary finding: Many fashionable large-scale improvements do not necessarily help under limited compute, while pipeline details and data handling matter a lot.
- Why it matters for `10x+` training efficiency: It gives us a methodology for separating commodity-useful tricks from frontier-lab cargo cult.
- Risk of overinterpreting the side effect: Masked LM and GLUE are not the same target as reasoning-capable autoregressive models.

## Hardware Reality

Extremely high. This is one of the few papers that treats single-GPU training as the primary design constraint rather than an afterthought.

## Composability

Likely composes with:

- DataComp-LM and DoReMi miniature curation loops.
- Optimizer-memory methods like GaLore/Q-GaLore.
- Local benchmark harness design.
- Small-model reasoning probes.

Likely conflicts with:

- Claims that depend on multi-node infrastructure.
- Methods whose overhead dominates at small batch or small model scale.
- Evaluation choices that cannot run cheaply and repeatedly.

## Reproduction Path

The repository gives a direct path:

- Install the package.
- Run the sanity-check pretraining step.
- Reproduce or shorten the crammed BERT recipe.
- Adapt the harness ideas to a tiny autoregressive baseline with explicit cost instrumentation.

The first use for us is likely not to train crammed BERT itself, but to borrow the constrained-budget experimental discipline.

## Verdict

- Evidence strength: High as a commodity-training methodology; medium for our exact autoregressive reasoning target.
- Estimated usable gain: Not a standalone multiplier; it is a baseline and harness discipline that prevents us from overvaluing methods that only work at frontier scale.
- Risk: We may learn more about BERT/GLUE optimization than LLM reasoning.
- Follow-up: Use Cramming to shape the RTX 4070 smoke-test protocol and baseline accounting.
