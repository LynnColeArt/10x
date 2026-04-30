# Replication Chatter: Sparse Training

## Scope

Sources reviewed:

- Structured RigL / condensed sparsity issues: https://github.com/calgaryml/condensed-sparsity/issues
- Early-Bird Tickets issues: https://github.com/GATECH-EIC/Early-Bird-Tickets/issues

## Main Signal

Sparse-training chatter is thinner than data/memory chatter, but it sharpens the hardware warning: the details of what gets sparsified, whether the structure is true `N:M`, and whether attention internals are covered matter enormously.

## Structured RigL Landmines

### Not Arbitrary N:M Sparsity

Structured RigL issue #64 asks how to adapt SRigL to `2:4` or `4:8` N:M sparsity. A maintainer says the repo implements constant fan-in, not arbitrary N:M, and points to JaxPruner for arbitrary N:M support: https://github.com/calgaryml/condensed-sparsity/issues/64

Implication:

- We should not assume SRigL maps directly to NVIDIA 2:4 sparse Tensor Core acceleration.
- Hardware-legible sparsity still needs a kernel-specific design pass.

### Transformer Weight Coverage Needs Care

Structured RigL issue #77 clarifies that SRigL can sparsify Q/K/V weights through PyTorch MHA `in_proj_weight`, but does not support sparsifying scaled dot-product attention itself: https://github.com/calgaryml/condensed-sparsity/issues/77

Implication:

- A Transformer sparse experiment should specify exactly which projections are sparse.
- Attention compute may remain dense even if projection weights are sparse.

### Custom Architectures Need Manual Integration

Structured RigL issue #78 asks about applying fixed fan-in to a custom two-layer MLP with binary-ish activations and predetermined fan-in: https://github.com/calgaryml/condensed-sparsity/issues/78

Implication:

- The framework is research-code flexible, not a drop-in optimizer for arbitrary models.

## Early-Bird Tickets Chatter

The Early-Bird repo has limited issue traffic. Open issues include a `resprune_50` concern and closed discussion around mask distance/thresholding: https://github.com/GATECH-EIC/Early-Bird-Tickets/issues

Implication:

- There is not much public stress-testing to lean on.
- We need to run our own mask-stability probe before treating early-bird behavior as portable.

## Experiment Impact

For sparse-training experiment cards, require:

- Explicit sparse target modules.
- Structured versus unstructured mask accounting.
- Kernel/backend plan.
- Dense attention versus sparse projection separation.
- Wall-clock measurement, not only parameter/FLOP sparsity.
- Mask-stability tracking over early training.

## Provisional Verdict

Sparse training remains a high-upside weird lane, but it is not mature enough to be our first `10x` stack. First experiment should be a small diagnostic mask-stability and sparse-kernel test, not a full training claim.
