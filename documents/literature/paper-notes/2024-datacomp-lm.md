# Paper Note: DataComp-LM

## Citation

- Title: DataComp-LM: In search of the next generation of training sets for language models
- Authors: Jeffrey Li et al.
- Year: 2024, revised 2025
- Link: https://arxiv.org/abs/2406.11794
- Code: https://github.com/mlfoundations/dclm
- Project: https://www.datacomp.ai/dclm/

## One-Sentence Claim

Controlled dataset design can substantially improve language-model quality per unit compute, making data curation one of the strongest near-term candidates for `10x` effective training efficiency.

## Efficiency Axis

- FLOPs: DCLM-Baseline is reported to achieve competitive quality with less training compute than several open baselines.
- Wall-clock: Reduced compute should reduce wall-clock at fixed hardware, though full-scale dataset processing can be expensive.
- Memory: No direct model-memory reduction.
- Energy/cost: Lower training compute implies lower cost; data processing cost must be counted.
- Data required: The framework starts from a huge Common Crawl pool, but the key intervention is filtering, deduplication, and mixing.
- Hidden teacher or preprocessing cost: Data extraction, filtering, deduplication, model-based filtering, distributed processing, and storage.

## Training Regime

- Pretraining: Yes, from-scratch language model pretraining.
- Continued pretraining: Not the main focus.
- Fine-tuning: No.
- Pruning: No.
- Inference-only: No.
- Evaluation-only: Provides a benchmark/evaluation suite, but the project includes training recipes.

## Baseline

The baseline framing is strong because DCLM fixes recipes and compares dataset construction strategies across model scales from roughly 412M to 7B parameters. It is closer to our actual objective than vision pruning papers because it directly targets language-model pretraining.

## Evidence

DCLM provides a standardized Common Crawl-derived corpus, OpenLM-based pretraining recipes, and more than 50 downstream evaluations. The arXiv abstract reports that DCLM-Baseline trains a 7B model from scratch to 64% 5-shot MMLU using 2.6T tokens. It reports a 6.6 percentage-point MMLU improvement over MAP-Neo while using 40% less compute, and similar average performance to Llama 3 8B across 53 natural language understanding tasks while using 6.6x less compute.

The repository includes processing, filtering, deduplication, tokenization, shuffling, training, and evaluation workflows. It also exposes a practical warning: serious dataset curation has its own compute, RAM, disk, and distributed-processing requirements.

## Capability Preservation

Capability is measured through MMLU and a broad 53-task NLU suite. This is much more relevant to our target than MNIST/CIFAR, though still not a full reasoning-product evaluation.

## Accidental Or Side-Effect Signal

- Original goal: Build a benchmark and controlled competition for language-model data curation.
- Unexpected or secondary finding: Dataset construction can rival or exceed many model-recipe changes at fixed compute.
- Why it matters for `10x+` training efficiency: It suggests a large fraction of training waste may live in token selection rather than architecture.
- Risk of overinterpreting the side effect: Full-scale curation may require large raw corpora, preprocessing infrastructure, and filtering compute that small labs do not have.

## Hardware Reality

The full DCLM pipeline is probably too large for casual local reproduction. However, the mechanism scales down well: we can run miniature data-selection experiments using small corpora and proxy models on the RTX 4070, then use DGX Spark for larger ablations.

## Composability

Likely composes with:

- Sparse training.
- Recurrent-depth models.
- Optimizer and memory-efficiency methods.
- Curriculum learning.
- Proxy-model data selection.
- Synthetic textbook/exercise generation.

Likely conflicts with:

- Claims that ignore preprocessing cost.
- Small experiments whose validation tasks are too narrow to catch data overfitting.
- Teacher-filtering approaches that rely on unavailable or expensive proprietary models.

## Reproduction Path

First local experiment should be a miniature DCLM-style curation loop:

- Select a small raw corpus.
- Build naive, deduped, filtered, and mixed variants.
- Train the same tiny Transformer for a fixed token/FLOP budget.
- Evaluate perplexity plus reasoning/task probes.
- Track preprocessing time separately from training time.

DGX Spark can later run a more serious 100M-1B parameter comparison if the miniature results show signal.

## Verdict

- Evidence strength: High for LLM pretraining relevance.
- Estimated usable gain: `2x-6x` effective compute is plausible from data curation alone, depending on baseline quality. It should not be counted as free because preprocessing and filtering cost can be large.
- Risk: Gains may depend on massive raw data pools and expensive filtering not available to small labs.
- Follow-up: Make this one of the first experiment cards. Data quality is likely the most realistic near-term multiplier in the `10x` stack.
