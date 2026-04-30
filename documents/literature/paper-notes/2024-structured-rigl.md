# Paper Note: Dynamic Sparse Training with Structured Sparsity

## Citation

- Title: Dynamic Sparse Training with Structured Sparsity
- Authors: Mike Lasby, Anna Golubeva, Utku Evci, Mihai Nica, Yani Ioannou
- Year: 2024, ICLR
- Link: https://arxiv.org/abs/2305.02299
- Code: https://github.com/calgaryml/condensed-sparsity

## One-Sentence Claim

Structured RigL learns sparse networks from sparse initialization while enforcing hardware-friendlier `N:M`-style structure, preserving generalization better than simpler structured sparse methods.

## Efficiency Axis

- FLOPs: Sparse-to-sparse training can reduce theoretical training compute.
- Wall-clock: The paper reports real acceleration for a 90% sparse linear layer, including `3.4x/2.5x` CPU acceleration versus dense/CSR and `1.7x/13.0x` GPU acceleration at batch size 256 versus dense/CSR.
- Memory: Sparse topology reduces active weights; structured representation is more efficient than arbitrary unstructured sparsity.
- Energy/cost: Not the primary measured claim, but wall-clock and sparsity imply potential savings.
- Data required: Same task data.
- Hidden teacher or preprocessing cost: Dynamic sparse topology updates and salience computation.

## Training Regime

- Pretraining: Not LLM pretraining.
- Continued pretraining: No.
- Fine-tuning: No, primarily training sparse models.
- Pruning: Not post-hoc pruning; dynamic sparse training.
- Inference-only: No, though hardware acceleration is shown for sparse layers.
- Evaluation-only: No.

## Baseline

The key baseline is unstructured dynamic sparse training such as RigL, plus dense and CSR sparse layer comparisons for acceleration. This is stronger than pure lottery-ticket evidence because it addresses the real hardware gap between theoretical sparsity and speed.

## Evidence

The paper proposes SRigL, a sparse-to-sparse dynamic sparse training method. It learns a hybrid fine-grained and structured sparse topology by enforcing a constant fan-in constraint and using neuron ablation at high sparsity. The repository supports MNIST, CIFAR-10, ImageNet ResNet-50/MobileNet, and ViT-B/16 configurations.

The strongest project-relevant evidence is not the vision accuracy itself; it is the explicit admission that unstructured sparsity is hard to accelerate, followed by a structured method with measured speedups on commodity-like dense/sparse layer comparisons.

## Capability Preservation

Capability preservation is measured through generalization on vision architectures, including CNNs and ViT. There is no LLM reasoning evidence yet.

## Accidental Or Side-Effect Signal

- Original goal: Make dynamic sparse training compatible with structured sparsity.
- Unexpected or secondary finding: The paper makes clear that sparsity has to be shaped for kernels, not just counted as zero weights.
- Why it matters for `10x+` training efficiency: It converts sparse-training evidence from abstract parameter reduction into hardware-legible training design.
- Risk of overinterpreting the side effect: Layer-level acceleration does not guarantee end-to-end LLM training acceleration at realistic batch sizes.

## Hardware Reality

This paper is highly relevant because it attacks the biggest practical weakness of the lottery-ticket lane: unstructured masks are often fake speedups. SRigL's `N:M`-style constant fan-in structure is closer to what real kernels can exploit.

## Composability

Likely composes with:

- Early-bird mask discovery.
- Low-precision training.
- Small Transformer experiments.
- Data-efficiency methods that reduce total tokens.
- Hardware-specific sparse kernels.

Likely conflicts with:

- Dense optimizer tricks that assume every parameter is updated every step.
- MoE routing if both mechanisms add topology instability.
- Architectures where the best sparse pattern is not constant-fan-in friendly.

## Reproduction Path

Start with the official repository on a vision task only if we need to validate the implementation. For project value, adapt the principle to a tiny Transformer MLP and/or attention projection:

- Compare dense, unstructured RigL, and structured constant-fan-in sparse training.
- Measure tokens/sec, memory, loss, and simple reasoning benchmarks.
- Check whether the speedup survives PyTorch/kernel reality on RTX 4070.

## Verdict

- Evidence strength: Medium-high for hardware-aware sparse training outside LLMs; medium-low for LLM transfer until tested.
- Estimated usable gain: Potentially `1.5x-3x` wall-clock if structure maps to kernels; higher theoretical sparsity should not be counted until measured.
- Risk: Language-model quality may degrade more sharply than vision quality, or sparse kernels may not accelerate at our target batch sizes.
- Follow-up: Pair with an Early-Bird tiny-LM mask-stability probe. This is the leading candidate for making lottery-ticket ideas hardware-legible.
