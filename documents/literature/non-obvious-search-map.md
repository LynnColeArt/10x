# Non-Obvious Search Map

The project should not only follow the obvious LLM-efficiency trail. If a `10x+` answer exists, it may sit at the intersection of fields that normally do not cite each other. This map keeps the search creative without letting it become unbounded.

## Search Rule

For each lateral lane, we ask:

- What training waste does this lane claim exists?
- Does it offer a mechanism, not just a metaphor?
- Can the mechanism be expressed as a small experiment?
- Does it reduce training cost, shift cost elsewhere, or only improve inference?
- What would make us stop pursuing it?

## Lateral Lanes

### Developmental Curricula

Human learning does not begin with internet-scale text. It begins with compressed, staged, embodied, and socially scaffolded data.

Search targets:

- child-language-inspired curricula
- synthetic developmental corpora
- simple-to-complex reasoning curricula
- textbook and exercise generation
- capability emergence in tiny models

Connection to `10x`:

Curriculum may reduce token waste by ordering examples so the model learns reusable primitives before high-entropy mixtures.

### Sparse Coding And Neural Circuits

Lottery-ticket work may be one instance of a broader sparse-circuit principle.

Search targets:

- sparse coding
- modular neural computation
- mechanistic circuit discovery
- supermasks
- low-rank capability localization
- feature superposition and disentanglement

Connection to `10x`:

If capabilities localize into sparse or low-rank circuits, training may be separable into circuit discovery and circuit fitting.

### Memory Consolidation And Replay

Current pretraining is mostly one long pass over shuffled text. Biological and continual-learning systems use replay, consolidation, and staged integration.

Search targets:

- replay buffers for language models
- sleep-like consolidation
- curriculum plus rehearsal
- continual pretraining without forgetting
- sample reweighting over training time

Connection to `10x`:

Selective replay could replace brute-force multi-epoch token exposure with targeted reinforcement of unstable or high-value concepts.

### Adaptive Compute And Halting

Not every token, example, layer, or training phase deserves equal compute.

Search targets:

- adaptive computation time
- token-level compute allocation
- layer skipping
- early exit
- looped Transformer halting
- dynamic depth during training

Connection to `10x`:

If easy examples or easy tokens can be trained with less compute, total training can shrink without changing the hard-case budget.

### Data Valuation And Active Selection

The data lane should include active selection methods, not just static filtering.

Search targets:

- influence functions
- gradient matching
- coreset selection
- active learning for language modeling
- data attribution
- proxy-model data scoring

Connection to `10x`:

A cheap proxy model may identify which examples are worth expensive training steps.

### Search, Self-Play, And Verifiers

Reasoning may not require learning every final behavior directly from text. It may require learning from generated problems, failed attempts, verifiers, and repair loops.

Search targets:

- self-play for language models
- verifier-guided training
- rejection sampling fine-tuning
- process supervision
- synthetic problem generation
- automated curriculum generation

Connection to `10x`:

Generated high-gradient tasks may replace low-density web tokens, but hidden teacher/verifier cost must be counted.

### Information Theory And Compression

Training may be spending too much compute encoding redundant information.

Search targets:

- minimum description length
- neural compression
- dataset deduplication
- mutual-information data filtering
- entropy-based pruning
- compression-based curriculum

Connection to `10x`:

If training data contains massive redundancy, compression-aware data selection could remove tokens without removing learning signal.

### Hardware-Legible Algorithms

Some methods are only efficient on paper. We need approaches that line up with real kernels and memory systems.

Search targets:

- `N:M` sparsity
- block sparsity
- FP8/FP4 training
- optimizer-state quantization
- activation checkpointing
- IO-aware attention and kernels
- AMD, Apple Silicon, and NVIDIA consumer constraints

Connection to `10x`:

A theoretical FLOP reduction only matters if wall-clock or memory limits improve on accessible hardware.

### Small-Model Reasoning Niches

The target is not necessarily a general frontier model. There may be niches where small models can learn reasoning primitives unusually efficiently.

Search targets:

- math/coding microcurricula
- theorem proving mini-domains
- algorithmic data
- tool-use-specialized models
- narrow-domain expert models

Connection to `10x`:

Specialized training can reveal mechanisms before we try broader language generalization.

### Thermodynamic And Analog Computing

This is probably not the first software path, but it may reveal alternate cost models.

Search targets:

- thermodynamic computing
- analog in-memory training
- energy-based learning
- equilibrium propagation
- neuromorphic sparse training

Connection to `10x`:

Even if hardware is not immediately usable, these fields may suggest algorithms that reduce digital training waste.

## Deliberate Weirdness Budget

For each major research milestone, reserve roughly `20%` of reading time for lateral sources that do not obviously belong to mainstream LLM training. The goal is not to become encyclopedic. The goal is to find mechanisms mainstream search would miss.

## Near-Term Weird Bets

- Recurrent-depth models plus early-ticket discovery.
- Proxy-model data selection plus dynamic sparse training.
- Developmental curriculum first, broad web continuation second.
- Sparse/low-rank capability localization as a guide for staged training.
- Adaptive loop count during training, not just inference.
