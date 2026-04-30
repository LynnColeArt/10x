# Replication Chatter: GaLore And Q-GaLore

## Scope

Sources reviewed:

- GaLore repo issue list and selected issues: https://github.com/jiaweizzhao/GaLore/issues
- Q-GaLore repo issue list and selected issues: https://github.com/VITA-Group/Q-GaLore/issues
- GaLore+ follow-up: https://arxiv.org/abs/2412.19820
- GaLore 2 follow-up: https://arxiv.org/abs/2504.20437
- Natural GaLore follow-up: https://arxiv.org/abs/2410.16029
- LocalLLaMA GaLore discussion: https://www.reddit.com/r/LocalLLaMA/comments/1bdk4z1/galore_a_training_strategy_that_allows_full/

## Main Signal

The memory-efficiency claim is real enough to keep prioritizing, but the chatter strongly reinforces our earlier caveat: GaLore/Q-GaLore are primarily **memory-accessibility** methods, not automatically training-speed methods.

## Landmines

### Memory Accounting Is Still Murky

GaLore issue #67 asks how the paper's estimated memory numbers should be computed after a user failed to match the table for LLaMA 350M: https://github.com/jiaweizzhao/GaLore/issues/67

Implication:

- We should not accept table memory numbers until we reproduce peak allocator memory, model memory, optimizer memory, and activation/checkpointing settings separately.

### SVD Can Fail Or Dominate Runtime

GaLore issue #58 reports SVD non-convergence during LLaMA-3 8B fine-tuning, with losses dropping to zero after the first batch before failure: https://github.com/jiaweizzhao/GaLore/issues/58

Follow-on papers also treat SVD overhead as a central weakness. GaLore+ says GaLore's SVD can consume more than `80%` of total training time in some settings and claims about `4x` fine-tuning speedup over vanilla GaLore by changing the projection approach: https://arxiv.org/abs/2412.19820

GaLore 2 explicitly lists SVD overhead and FSDP/parallelism integration as remaining challenges: https://arxiv.org/abs/2504.20437

Implication:

- Our first reproduction must log projection/SVD time separately from forward/backward/optimizer time.
- If SVD dominates, GaLore is a memory hack, not a `10x` training-efficiency ingredient unless paired with a faster projection method.

### Q-GaLore Repo Issues Suggest Implementation Immaturity

Q-GaLore issue #9 asks why the pretraining script uses distributed/NCCL constructs for a claimed single-GPU setting: https://github.com/VITA-Group/Q-GaLore/issues/9

Q-GaLore issue #10 reports a missing `absmax2` argument in the 8-bit optimizer path: https://github.com/VITA-Group/Q-GaLore/issues/10

Q-GaLore issue #11 reports checkpoint save failure from a non-picklable optimizer hook during Mistral-7B pretraining: https://github.com/VITA-Group/Q-GaLore/issues/11

Implication:

- Q-GaLore is too important to ignore but too fragile to trust from paper claims alone.
- Start with small official scripts and checkpoint/resume tests before attempting larger runs.

### Community Reaction Separates Fit From Feasible Training Time

A LocalLLaMA commenter reports successfully starting full-parameter training on C4 on an RTX 3090 at about `22.7GB` VRAM, but with estimated full-run time around months: https://www.reddit.com/r/LocalLLaMA/comments/1bdk4z1/galore_a_training_strategy_that_allows_full/

Implication:

- "Fits in VRAM" is a necessary condition for commodity training, not a sufficient condition.
- We need wall-clock-to-target-loss measurements, not just memory snapshots.

## Positive Signals

- Follow-on work exists quickly, which suggests the core idea has enough traction to attract extension rather than immediate abandonment.
- Natural GaLore claims improved optimization without extra memory overhead on 60M-1.1B LLaMA-style models: https://arxiv.org/abs/2410.16029
- GaLore 2 reports scaling to Llama 7B pretraining up to 500B tokens, which makes the lane more relevant than a toy memory trick if reproduced: https://arxiv.org/abs/2504.20437

## Experiment Impact

For any GaLore/Q-GaLore experiment card, require:

- Peak memory by category when possible.
- Tokens/sec.
- Projection/SVD time.
- Wall-clock to fixed validation loss.
- Checkpoint/save/resume test.
- AdamW and 8-bit AdamW baselines.
- No credit for memory savings as training-speed savings unless wall-clock improves.

## Provisional Verdict

Keep as a first-wave memory-access experiment, but downgrade any direct `10x` contribution until wall-clock and SVD overhead are measured locally.
