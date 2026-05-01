# Paper Note: DeepSeek-V4

## Citation

- Title: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- Authors: DeepSeek-AI
- Year: 2026
- Link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- Code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference

## One-Sentence Claim

DeepSeek-V4 combines hybrid compressed/sparse attention, MoE active-parameter scaling, mHC residual stabilization, Muon optimization, curriculum, precision placement, and custom systems work to make `1M` context practical while preserving frontier-level model capability.

## Efficiency Axis

- FLOPs: Strong for long-context inference. The report says V4-Pro uses `27%` of DeepSeek-V3.2 single-token inference FLOPs at `1M` context, while V4-Flash uses `10%`.
- Wall-clock: The report claims `1.50x-1.73x` speedup for general inference workloads from the MegaMoE fused kernel versus non-fused baselines, and up to `1.96x` for latency-sensitive rollouts/agent serving. It does not provide a clean public end-to-end pretraining wall-clock multiplier.
- Memory: Strong for KV cache. The report says V4-Pro uses `10%` and V4-Flash uses `7%` of DeepSeek-V3.2 KV cache at `1M` context. Against a BF16 GQA8 baseline, the report says DeepSeek-V4 KV cache can fall to about `2%` in the `1M` context setting.
- Energy/cost: Likely strong for long-context serving and RL rollouts; unclear for from-scratch training because the pretraining runs still use `32T-33T` tokens and very large infrastructure.
- Data required: No evidence of a data-volume reduction. V4-Flash is trained on `32T` tokens and V4-Pro on `33T` tokens.
- Hidden teacher or preprocessing cost: High. The result depends on large-scale data, post-training, domain experts, on-policy distillation, custom kernels, deterministic infrastructure, and specialized parallelism.

## Training Regime

- Pretraining: Yes, from-scratch frontier-scale pretraining.
- Continued pretraining: Not the main claim.
- Fine-tuning: Post-training pipeline includes SFT/RL-style domain expert cultivation and consolidation.
- Pruning: No.
- Inference-only: Some of the clearest efficiency claims are inference and KV-cache claims.
- Evaluation-only: No.

## Baseline

The main efficiency baseline is DeepSeek-V3.2, with additional discussion against a BF16 GQA8 attention baseline for KV-cache size. These are strong frontier-scale baselines, but they are not commodity baselines. We should not transfer the published `3.7x` or `10x` long-context serving ratios into our training-efficiency ledger without local experiments.

## Evidence

The official report describes two models:

- DeepSeek-V4-Pro: `1.6T` total parameters, `49B` activated parameters.
- DeepSeek-V4-Flash: `284B` total parameters, `13B` activated parameters.

Both support `1M` context. The architecture retains Transformer, DeepSeekMoE, and multi-token prediction components while adding:

- Hybrid attention with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA).
- Manifold-Constrained Hyper-Connections (mHC).
- Muon optimizer for most modules.
- FP4/FP8/BF16 precision placement.
- Curriculum that starts at `4K` sequence length and gradually extends to `16K`, `64K`, and `1M`.
- Sparse-attention warmup after an initial dense-attention phase.
- Anticipatory routing and SwiGLU clamping to mitigate MoE-related loss spikes.
- Custom systems work including MegaMoE, TileLang fused kernels, deterministic kernels, hybrid ZeRO for Muon, tensor-level checkpointing, and contextual parallelism.

The supplied video transcript is useful as a mechanism index, but the official report and API release note should remain the authority for quantitative claims.

## Capability Preservation

The official release and report claim strong world knowledge, reasoning, coding, math, agent, and long-context results. For this project, the important point is narrower: DeepSeek reports large long-context efficiency gains while still presenting the model as capability-preserving relative to prior DeepSeek models and competitive with frontier systems. Independent replication and public benchmark scrutiny still matter.

## Accidental Or Side-Effect Signal

- Original goal: Make million-token context practical in a frontier-scale MoE model.
- Unexpected or secondary finding: Several training-efficiency clues appear as stability and systems side effects: loss spikes tied to MoE routing/outliers, anticipatory routing, SwiGLU clamping, query/KV RMSNorm, sparse-attention warmup, short-to-long curriculum, deterministic kernels, and recomputation/fusion to keep mHC overhead bounded.
- Why it matters for `10x+` training efficiency: Failed runs, rollbacks, routing instability, non-deterministic kernels, and overly dense attention are hidden compute taxes. Removing those taxes may compound with data efficiency and memory-efficient optimization even if no single DeepSeek component is a `10x` training recipe.
- Risk of overinterpreting the side effect: Many reported gains are only visible at frontier scale or during long-context inference. Commodity hardware may not have the same bottlenecks, and custom kernels may be unavailable or slower at small batch sizes.

## Hardware Reality

The full models are not commodity-training targets. The relevant question is whether their mechanisms survive scale-down:

- Muon can be compared against AdamW in tiny Transformer pretraining.
- CSA/HCA ideas can be prototyped as long-context attention modules and profiled at `4K-64K` context.
- mHC can be tested as a stability mechanism in deep or looped tiny models.
- Anticipatory routing and SwiGLU clamping can be tested in tiny MoE training under intentionally spiky routing conditions.
- Curriculum from short to long contexts can be tested on the RTX 4070 before using DGX Spark time.

## Composability

Likely composes with:

- Data curation and curriculum.
- Muon or other optimizer-efficiency lanes.
- Low-precision training/inference.
- MoE active-parameter scaling.
- Recurrent-depth experiments, especially if mHC improves repeated-depth stability.
- Sparse/structured attention experiments.

Likely conflicts with:

- Simple commodity stacks that lack custom kernels.
- Any experiment where long context is not the bottleneck.
- Small-batch local training where communication overlap and expert parallelism are irrelevant.
- Overly broad `10x` accounting that mixes inference KV savings with pretraining FLOP savings.

## Reproduction Path

Start with small, isolatable experiments:

- **Optimizer:** Train a tiny Transformer with AdamW versus Muon under matched tokens, wall-clock, and memory.
- **Attention:** Implement a toy CSA/HCA-style attention block and compare dense attention, compressed attention, and sparse compressed retrieval at increasing sequence lengths.
- **Stability:** Add mHC-like constrained residual mixing to a deep or looped tiny model and track loss spikes, gradient norms, and wall-clock overhead.
- **Routing:** Train a tiny MoE with current routing versus lagged/anticipatory routing and SwiGLU clamping; deliberately stress it with skewed data mixtures.
- **Curriculum:** Compare full-length training from step zero versus `4K -> 16K -> 64K` style curriculum at a local scale.

## Verdict

- Evidence strength: High as an official frontier systems report; medium as direct evidence for commodity training efficiency.
- Estimated usable gain: Near-term `1.2x-3x` training-side gains may be plausible from selected components; the larger `3.7x-10x` claims are long-context inference/KV-cache gains, not from-scratch training gains.
- Risk: Very high risk of false accounting if we mix deployment efficiency, inference FLOPs, and training compute.
- Follow-up: Write standalone notes for Muon, mHC, and DSA, then turn the smallest two mechanisms into RTX 4070 experiment cards.

