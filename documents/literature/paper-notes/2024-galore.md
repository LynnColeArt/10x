# Paper Note: GaLore

## Citation

- Title: GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
- Authors: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
- Year: 2024, ICML Oral
- Link: https://arxiv.org/abs/2403.03507
- Code: https://github.com/jiaweizzhao/GaLore

## One-Sentence Claim

Projecting gradients into low-rank subspaces can reduce optimizer-state memory while preserving full-parameter learning for LLM pretraining and fine-tuning.

## Efficiency Axis

- FLOPs: Does not primarily reduce training FLOPs; projection updates add overhead.
- Wall-clock: May improve feasibility but not necessarily speed; SVD/projection can add cost.
- Memory: The abstract reports up to `65.5%` optimizer-state memory reduction; 8-bit GaLore reports up to `82.5%` optimizer-memory and `63.3%` total training-memory reduction versus BF16 baseline.
- Energy/cost: Lower memory can reduce hardware class needed; cost gain is mostly accessibility, not pure compute reduction.
- Data required: Same data.
- Hidden teacher or preprocessing cost: None, but projection-update overhead must be measured.

## Training Regime

- Pretraining: Yes, LLaMA-style models on C4.
- Continued pretraining: Not primary.
- Fine-tuning: Yes, RoBERTa on GLUE.
- Pruning: No.
- Inference-only: No.
- Evaluation-only: No.

## Baseline

Baselines include full-rank training, LoRA-style low-rank adaptation, and 8-bit optimizer variants. The key claim is stronger than LoRA for our purposes because it preserves full-parameter learning rather than freezing most weights.

## Evidence

The arXiv abstract reports memory savings while maintaining performance for LLaMA `1B` and `7B` architectures trained on C4 with up to `19.7B` tokens, plus RoBERTa fine-tuning on GLUE. It also claims feasibility of pretraining a `7B` model on a consumer `24GB` RTX 4090-class GPU.

Important caveat: the arXiv abstract says this is possible without model parallelism, checkpointing, or offloading, while the public repository's 7B single-GPU command uses activation checkpointing and reports `22.8G` memory. We should treat this as a reproduction/accounting detail to verify, not a reason to discard the method.

## Capability Preservation

Capability is measured through LM loss/downstream performance for pretraining and GLUE fine-tuning. It is relevant, but we need reasoning probes for our target.

## Accidental Or Side-Effect Signal

- Original goal: Reduce optimizer-state memory for LLM training.
- Unexpected or secondary finding: Full-parameter pretraining starts to fit into consumer GPU memory classes.
- Why it matters for `10x+` training efficiency: Accessibility may improve by a hardware-class jump even if FLOPs do not drop.
- Risk of overinterpreting the side effect: Memory feasibility is not the same as faster or cheaper end-to-end training.

## Hardware Reality

Very high relevance. GaLore is directly about making training fit on commodity or prosumer devices. The risk is that fitting a model into memory can still be too slow to be useful.

## Composability

Likely composes with:

- DataComp-LM and DoReMi.
- Q-GaLore / quantized optimizer states.
- Cramming-style local training harness.
- Recurrent-depth parameter reuse.

Likely conflicts with:

- Optimizers whose dynamics depend on full-rank states.
- Sparse training methods if gradient projection and sparse topology updates fight.
- Claims that require large batch sizes unavailable on small hardware.

## Reproduction Path

Start with 60M/130M LLaMA-style runs from the repository:

- Compare AdamW, 8-bit AdamW, GaLore, and 8-bit GaLore.
- Measure memory, tokens/sec, wall-clock to fixed loss, and eval quality.
- Only then test whether larger models become feasible on RTX 4070 or DGX Spark.

## Verdict

- Evidence strength: High for memory-efficiency relevance; medium for true training-efficiency gain.
- Estimated usable gain: Potentially `2x-4x` hardware accessibility via memory reduction; direct wall-clock gain is unknown and may be negative without careful kernels/settings.
- Risk: Projection overhead and checkpointing can hide the real cost.
- Follow-up: Promote to first-wave memory experiment card, but score memory and wall-clock separately.
