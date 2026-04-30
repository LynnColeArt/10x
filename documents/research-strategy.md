# Research Strategy

## Core Hypothesis

Modern LLM training is plausibly `10x+` more expensive than necessary for many useful reasoning-capable models because the dominant training recipe is optimized for frontier-lab scaling conditions rather than commodity-hardware learning efficiency.

The likely answer is not one isolated breakthrough. It is more likely a compatible stack of smaller discoveries across data quality, architecture, sparsity, recurrence, optimization, precision, and evaluation.

## Target Claim

Can we identify and validate a recipe for training or adapting reasoning-capable small LLMs at least `10x` more efficiently than a conventional dense Transformer baseline, using commodity-accessible hardware?

Efficiency must be tracked across:

- Training FLOPs
- Wall-clock time
- GPU memory
- Energy or approximate hardware cost
- Capability retained versus baseline
- Hidden cost, especially teacher-model inference or synthetic-data generation

## Research Posture

The project should be curious but adversarial. We should actively look for overlooked combinations, but every paper, repo, and claim needs to pass through the same evidence filter.

Important guardrails:

- Do not treat benchmark-only gains as capability preservation.
- Do not count the same efficiency gain twice under different names.
- Do not ignore hardware reality. Unstructured sparsity is not a speedup unless the target hardware benefits.
- Do not let teacher-model distillation hide the real cost unless we explicitly account for it.
- Do not assume dense frontier-lab pretraining is the only route to reasoning capability.
- Look for accidents and side effects. Some of the most important clues may appear as secondary findings in work that was nominally about alignment, stability, dataset benchmarking, pruning, deployment, or memory limits.

## Initial Research Lanes

### 1. Lottery Tickets And Sparse Training

Starting clue: dense training may be an expensive way to discover a much smaller effective subnetwork.

Questions:

- Can useful subnetworks be identified early enough to avoid most dense training?
- Do early-bird tickets hold for Transformer-scale language models?
- Which sparse methods map to real hardware speedups?
- Can dynamic sparse training or structured sparsity replace prune-after-training workflows?

Likely keywords:

- Lottery Ticket Hypothesis
- Early-bird tickets
- Dynamic sparse training
- RigL
- SET
- SNIP
- GraSP
- structured sparsity
- N:M sparsity
- sparse Transformers

### 2. Recurrent Depth And Parameter Reuse

Starting aggregate: OpenMythos is useful as a map of claims, not as evidence by itself.

Research focus:

- Recurrent-Depth Transformers
- Looped Transformers
- Universal Transformers
- parameter sharing across depth
- adaptive compute
- latent reasoning loops
- recurrent blocks with input injection

Key question:

Can a small recurrent-depth model plus adaptive loop count achieve reasoning behavior that would otherwise require a much deeper dense Transformer?

### 3. Data Efficiency

Data may be the highest-leverage efficiency source because many training tokens are redundant, low-signal, or actively harmful.

Questions:

- How much training compute is wasted on duplicate or low-quality tokens?
- Which filtering methods preserve reasoning and generalization?
- Can curriculum and hard-example mining reduce total token requirements?
- When does synthetic data genuinely help rather than overfit the teacher?

### 4. Distillation And Bootstrapping

Distillation is probably part of the answer, but it can also hide costs.

Questions:

- Which teacher signals are most compute-efficient?
- Are reasoning traces, verifiers, preference data, or final answers the most useful target?
- Can bootstrapping reduce dependency on proprietary frontier teachers?
- How do we account for teacher inference cost honestly?

### 5. Optimizer, Precision, And Memory Efficiency

Commodity hardware is often memory-bound before it is compute-bound.

Questions:

- Which optimizer-state reductions preserve pretraining quality?
- Can low-precision training reduce memory and wall-clock cost without destabilizing small-model training?
- Which methods are practical on NVIDIA consumer GPUs, DGX Spark-class hardware, AMD AI appliances, and Apple Silicon?

### 6. Architecture Alternatives

Some efficiency may come from leaving the standard dense Transformer stack.

Candidates:

- MoE and sparse routed models
- Mamba/state-space models
- RWKV-like recurrent models
- RetNet-style recurrence
- efficient attention variants
- hybrid recurrent-attention models

## Evidence Workflow

Each paper or project gets a standardized note in `documents/literature/paper-notes/`.

Every note should answer:

- What claim is being made?
- What baseline is used?
- Is the work about pretraining, continued pretraining, fine-tuning, pruning, inference, or evaluation?
- What model sizes and datasets were tested?
- What capability was preserved, improved, or lost?
- Was the efficiency clue the paper's original goal, or an accidental side effect?
- Is the efficiency measured in FLOPs, memory, wall-clock, energy, or something else?
- Does the method save real time on real hardware?
- Does it compose with other methods?
- What does it conflict with?
- Is there reproducible code?
- Could we test a reduced version on an RTX 4070 or DGX Spark?

## Synthesis Workflow

The synthesis phase should produce candidate stacks, not just a bibliography.

Candidate stack examples:

- Sparse Ticket Stack: early ticket discovery + dynamic structured sparsity + low precision.
- Recurrent Depth Stack: looped Transformer + adaptive compute + parameter sharing + compact data curriculum.
- Data-First Stack: aggressive data filtering + curriculum + synthetic reasoning data + small dense baseline.
- Memory-Efficient Stack: dense small model + optimizer-state reduction + quantized training + activation memory controls.
- Hybrid Stack: recurrent or sparse architecture + targeted distillation + verifier-guided training.

Each stack needs:

- Expected efficiency gain
- Main evidence
- Compatibility risks
- Hardware assumptions
- Baseline comparison
- Experimental proof path
- Failure criteria

## Milestones

### Milestone 1: Research Dossier

Deliverables:

- Literature map across 2022-2026
- Evidence matrix
- OpenMythos source-map extraction
- Lottery-ticket and sparse-training memo
- Initial candidate stacks

Exit criteria:

- At least three experiment-ready hypotheses
- Clear baseline definition
- Known hardware constraints documented

### Milestone 2: Baseline Harness

Deliverables:

- Small dense Transformer baseline
- Dataset and tokenizer choice
- Training cost instrumentation
- Evaluation suite for reasoning, language quality, and regression checks

Exit criteria:

- Baseline can run on RTX 4070 for smoke tests
- Main runs can target DGX Spark
- Metrics capture wall-clock, memory, tokens, and eval quality

### Milestone 3: Candidate Experiments

Deliverables:

- Experiment cards for top candidate stacks
- Small-scale ablations
- Failure notes and revised hypotheses

Exit criteria:

- At least one candidate shows credible movement toward `10x`
- Or we can explain, with evidence, why the attempted stack fails

## OpenMythos Handling

OpenMythos should be treated as a useful research aggregate and hypothesis generator.

Rules:

- Do not treat OpenMythos as evidence that Anthropic uses a specific architecture.
- Extract its cited sources and testable architectural claims.
- Score recurrent-depth claims independently from any Claude Mythos speculation.
- Track whether recurrent depth improves training efficiency, inference-time reasoning, or both.

## Current Bet

The most promising near-term path is probably not pure lottery-ticket pruning. It is more likely a hybrid of:

- Early or dynamic discovery of efficient computation paths
- Hardware-legible sparsity or parameter reuse
- Higher-signal training data
- Memory-aware training techniques
- Evaluation that rewards real reasoning rather than benchmark memorization
