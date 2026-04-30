# Replication Chatter: Cramming

## Scope

Sources reviewed:

- Cramming repo issue list and selected issues: https://github.com/JonasGeiping/cramming/issues
- Reddit MachineLearning discussion: https://www.reddit.com/r/MachineLearning/comments/zy8c99
- Reddit MLScaling discussion: https://www.reddit.com/r/mlscaling/comments/1ef426n

## Main Signal

Cramming remains valuable as a methodology for commodity training, but issue chatter shows why fixed-budget experiments are treacherous: torch versions, compile settings, data exhaustion, tokenized dataset choice, and microbatch saturation all change results.

## Landmines

### Default Reproduction Was Initially Fragile On Newer Hardware/Software

Cramming issue #45 reports that a user got only about `0.73` GLUE after 24h on a single A100 40GB, below paper expectations. The maintainer investigated and pointed to data wraparound, `torch.compile` settings, modern package versions, and tokenized dataset size: https://github.com/JonasGeiping/cramming/issues/45

Later in the same issue, another user reports GLUE around `80.3` after using a larger dataset slice and settings guidance.

Implication:

- Cramming is reproducible enough to learn from, but not plug-and-play.
- Our baseline harness must pin environment versions, data shards, and compile settings.

### GPU Utilization Is Part Of The Recipe

In issue #45, a maintainer clarifies that the default microbatch size is conservative so the code runs on many GPUs, but users should increase it to saturate their card. In 24h-budget mode, better GPU utilization improves final quality even if fixed-token results would be similar: https://github.com/JonasGeiping/cramming/issues/45

Implication:

- Hardware-aware tuning is not optional for commodity experiments.
- Experiment cards need both fixed-token and fixed-wall-clock modes.

### Apple Silicon Was Not A First-Class Target

Cramming issue #36 reports a Mac M1 illegal hardware instruction; the maintainer suspects ARM/x86_64 dependency mismatch and notes it is hard to debug without M1 access: https://github.com/JonasGeiping/cramming/issues/36

Implication:

- MacBook feasibility needs its own stack, not assumptions carried from CUDA.

### Consumer GPU Feasibility Is Real But Narrow

Cramming issue #26 asks whether an RTX 3060 12GB can pretrain a BERT model; the maintainer's answer is essentially yes, that is the repository's point: https://github.com/JonasGeiping/cramming/issues/26

Implication:

- Cramming validates the commodity-training frame, but only for a specific masked-LM recipe.

## External Chatter

The Reddit MachineLearning discussion praises the single-GPU benchmark format and the paper's failed-attempt reporting, but also raises a core question: whether 24-hour training behavior predicts long-run training behavior or downstream value: https://www.reddit.com/r/MachineLearning/comments/zy8c99

The MLScaling discussion includes skepticism about interpreting the paper as a massive acceleration over historical BERT compute: https://www.reddit.com/r/mlscaling/comments/1ef426n

Implication:

- Cramming should shape our experimental discipline, not be treated as evidence that LLM pretraining is already `1000x` cheaper.

## Experiment Impact

For our baseline harness, require:

- Fixed-wall-clock and fixed-token modes.
- Environment lockfile.
- Compile settings recorded.
- Microbatch and GPU utilization reported.
- Data wraparound/epoch count logged.
- Separate masked-LM and autoregressive baselines.

## Provisional Verdict

Cramming is not a direct recipe for reasoning-capable autoregressive LLMs, but it is currently the best methodological guardrail for our RTX 4070 smoke tests.
