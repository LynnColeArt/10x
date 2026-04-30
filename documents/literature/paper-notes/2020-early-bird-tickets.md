# Paper Note: Drawing Early-Bird Tickets

## Citation

- Title: Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks
- Authors: Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, Yingyan Lin
- Year: 2020, ICLR
- Link: https://openreview.net/forum?id=BJxsrgStvr
- Code: https://github.com/GATECH-EIC/Early-Bird-Tickets

## One-Sentence Claim

Winning-ticket masks can stabilize early in training, enabling a model to identify and continue training a sparse subnetwork before paying the full train-prune-retrain cost.

## Efficiency Axis

- FLOPs: Potentially reduced by switching to the early-bird subnetwork during training.
- Wall-clock: Intended to reduce training time, though exact wall-clock depends on sparse implementation.
- Memory: Reduced after pruning to the EB ticket.
- Energy/cost: The OpenReview abstract reports up to `5.8x-10.7x` energy savings while maintaining comparable or better accuracy.
- Data required: Same supervised datasets as baseline.
- Hidden teacher or preprocessing cost: Mask-distance tracking and early search overhead.

## Training Regime

- Pretraining: No LLM pretraining.
- Continued pretraining: No.
- Fine-tuning: Pruned subnetworks are continued or retrained in supervised vision settings.
- Pruning: Yes, with early mask detection.
- Inference-only: No.
- Evaluation-only: No.

## Baseline

The baseline is conventional full training plus pruning/retraining in vision models. This is a credible baseline for pruning research, but the evidence does not directly cover autoregressive language modeling.

## Evidence

The paper argues that useful masks become stable early and proposes a mask-distance metric to detect that stabilization without knowing the final winning ticket. It tests multiple deep networks and datasets and reports energy savings in the `5.8x-10.7x` range in the OpenReview abstract. The associated repository includes CIFAR and ImageNet-oriented workflows for VGG and ResNet-style models, plus low-precision search/retrain variants.

One documentation wrinkle: the public repository README mentions up to `4.7x` energy savings, while the OpenReview abstract reports `5.8x-10.7x`. We should rely on the paper for claims but preserve this discrepancy as a reproduction warning.

## Capability Preservation

Capability preservation is classification accuracy on vision datasets. There is no evidence here about LLM perplexity, instruction following, or reasoning preservation.

## Accidental Or Side-Effect Signal

- Original goal: Reduce pruning and retraining cost for deep networks.
- Unexpected or secondary finding: Useful masks can stabilize far earlier than full convergence.
- Why it matters for `10x+` training efficiency: This is the point where lottery-ticket thinking starts becoming a possible training-time saving rather than only post-hoc compression.
- Risk of overinterpreting the side effect: Early mask stability in CNNs may not transfer to autoregressive Transformers or reasoning-capable language models.

## Hardware Reality

More practical than the original lottery-ticket workflow because it tries to avoid dense discovery cost. However, whether it produces real speedups for LLMs depends on the sparsity structure and kernel support. Channel pruning in CNNs is more hardware-friendly than arbitrary Transformer weight sparsity.

## Composability

Likely composes with:

- Structured sparsity.
- Low-precision training.
- Dynamic sparse training.
- Proxy-model architecture search.
- Small language-model smoke tests.

Likely conflicts with:

- Training recipes where early masks are unstable.
- Highly adaptive architectures where early sparsity prematurely removes late-useful capacity.
- Unstructured sparse kernels that do not accelerate on target hardware.

## Reproduction Path

A reduced reproduction can run on RTX 4070 using the public code for CIFAR. For our project, the more valuable adaptation is a tiny Transformer experiment:

- Train a dense tiny LM for a short warmup.
- Track candidate structured masks over early checkpoints.
- Detect mask stabilization.
- Continue sparse training.
- Compare tokens-to-loss, wall-clock, and reasoning proxy tasks against dense baseline.

## Verdict

- Evidence strength: Medium-high for vision training efficiency; unknown for LLMs.
- Estimated usable gain: Potentially `2x-5x` if early stable masks transfer to Transformers and map to real kernels. The headline `5.8x-10.7x` should be treated as a vision-domain upper signal, not assumed for LLMs.
- Risk: Early structure may not stabilize in small autoregressive Transformers, or may be too unstructured to accelerate.
- Follow-up: This is the first sparse-lane candidate for an experiment card after a tiny-LM mask-stability probe.
