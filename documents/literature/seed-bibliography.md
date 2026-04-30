# Seed Bibliography

This is the first source corpus for the `10x+` training-efficiency investigation. It is intentionally broad enough to catch non-obvious paths, but every source still needs a structured paper note before we treat it as evidence.

## Source Tiers

- **Tier 1:** directly relevant to training-efficiency claims we may test.
- **Tier 2:** relevant mechanism, but likely indirect or needs adaptation.
- **Tier 3:** hypothesis generator, aggregate, or lateral inspiration.

## Anchor Lane: Lottery Tickets And Sparse Training

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | 1 | https://arxiv.org/abs/1803.03635 | Establishes the core clue that dense networks can contain much smaller trainable subnetworks. | Can the ticket be discovered early enough to save training compute? |
| Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks | 1 | https://openreview.net/forum?id=BJxsrgStvr | Claims winning connectivity patterns can be identified early in training. | Does early-bird behavior survive Transformer language-model training? |
| Rigging the Lottery: Making All Tickets Winners | 2 | https://arxiv.org/abs/1911.11134 | RigL grows and prunes sparse connectivity during training. | Is dynamic sparsity practical for autoregressive LMs on our hardware? |
| Dynamic Sparse Training with Structured Sparsity | 1 | https://arxiv.org/abs/2305.02299 | Moves dynamic sparse training toward structured `N:M` sparsity that hardware can accelerate. | Can structured dynamic sparsity become a real wall-clock gain? |
| SNIP: Single-shot Network Pruning based on Connection Sensitivity | 2 | https://arxiv.org/abs/1810.02340 | Finds sparse subnetworks at initialization before training. | Is single-shot selection too weak for LLMs, or useful as a warm-start signal? |
| GraSP: Gradient Signal Preservation | 2 | https://arxiv.org/abs/2002.07376 | Another early pruning criterion focused on preserving gradient flow. | Does gradient-preservation scoring correlate with small-LM training success? |
| Lottery Tickets can have Structural Sparsity | 2 | https://openreview.net/forum?id=oZe7Zdia1H5 | Connects lottery tickets to structured pruning rather than arbitrary unstructured masks. | Can structural tickets bridge the paper-speedup versus hardware-speedup gap? |
| SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot | 2 | https://arxiv.org/abs/2301.00774 | Shows large pretrained GPT-family models can be pruned without retraining. | Is this only inference compression, or does it reveal trainable sparse structure? |
| Wanda: A Simple and Effective Pruning Approach for Large Language Models | 2 | https://arxiv.org/abs/2306.11695 | Weight-and-activation pruning without retraining. | Can activation-informed pruning become an early training signal? |
| LLM-Pruner: On the Structural Pruning of Large Language Models | 2 | https://arxiv.org/abs/2305.11627 | Structured LLM pruning lane. | Which structures can be removed while preserving reasoning? |

## Anchor Lane: Recurrent Depth And Parameter Reuse

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| OpenMythos | 3 | https://github.com/kyegomez/OpenMythos | Useful aggregate of recurrent-depth claims and citations, not primary evidence. | Which cited claims are testable without accepting the Claude Mythos framing? |
| Parcae: Scaling Laws For Stable Looped Language Models | 1 | https://arxiv.org/abs/2604.12946 | Current looped-language-model scaling-law work; directly relevant to parameter reuse. | Does looping reduce training cost, or mainly improve parameter efficiency/inference scaling? |
| Universal Transformers | 2 | https://arxiv.org/abs/1807.03819 | Older recurrent-depth Transformer formulation with adaptive computation ideas. | What did Universal Transformers get right that current LLM recipes ignored? |
| Reasoning with Latent Thoughts: On the Power of Looped Transformers | 1 | https://arxiv.org/abs/2502.17416 | Links looped Transformers to latent reasoning and effective depth. | Can latent loops substitute for deeper static stacks on reasoning tasks? |
| Training Large Language Models to Reason in a Continuous Latent Space | 1 | https://arxiv.org/abs/2412.06769 | Coconut-style latent reasoning may shift reasoning from token space into hidden states. | Is this a training-efficiency trick, an inference-efficiency trick, or both? |
| Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA | 1 | https://arxiv.org/abs/2410.20672 | Parameter sharing across depth with layer-wise LoRA relaxation. | Can we initialize or train compact recursive models without inheriting a huge pretrained cost? |

## Data Efficiency And Curriculum

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| DataComp-LM: In search of the next generation of training sets for language models | 1 | https://arxiv.org/abs/2406.11794 | Strong evidence that dataset design can yield large compute-equivalent gains. | Which curation steps are cheap enough for commodity researchers? |
| DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | 1 | https://arxiv.org/abs/2305.10429 | Uses proxy models to tune data mixtures and reports fewer training steps to target quality. | Can small local proxy models choose data mixtures for larger local runs? |
| Textbooks Are All You Need | 1 | https://arxiv.org/abs/2306.11644 | High-quality textbook-like and synthetic data can produce strong small models in code. | How much of the gain is data quality versus hidden teacher cost? |
| TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | 2 | https://arxiv.org/abs/2305.07759 | Shows tightly scoped synthetic data can teach surprisingly small models coherent behavior. | Can developmental curricula scale from toy language to reasoning primitives? |
| Scaling Data-Constrained Language Models | 2 | https://arxiv.org/abs/2305.16264 | Studies regimes where data is scarce or reused. | Can deliberate data augmentation beat brute-force token volume? |
| Cramming: Training a Language Model on a Single GPU in One Day | 1 | https://arxiv.org/abs/2212.14034 | Directly aligned with commodity-hardware training constraints. | Which recipe choices transfer from masked LM cramming to autoregressive reasoning models? |

## Distillation, Bootstrapping, And Alignment Efficiency

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| Orca: Progressive Learning from Complex Explanation Traces of GPT-4 | 2 | https://arxiv.org/abs/2306.02707 | Rich teacher traces improve small-model reasoning compared with shallow imitation. | How do we account for teacher inference cost honestly? |
| LIMA: Less Is More for Alignment | 2 | https://arxiv.org/abs/2305.11206 | Meta work showing strong alignment behavior from only carefully curated examples after pretraining. | Does this inform training efficiency, or only post-training leverage? |
| Long Is More for Alignment | 2 | https://arxiv.org/abs/2402.04833 | Challenges LIMA-style data selection with a simple long-instruction baseline. | Which “small data” effects are robust versus benchmark artifacts? |

## Optimizer, Precision, And Memory Efficiency

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training | 1 | https://arxiv.org/abs/2305.14342 | Claims faster convergence for LM pretraining than Adam-style baselines. | Does the speedup persist in small-model commodity settings? |
| GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | 1 | https://arxiv.org/abs/2403.03507 | Full-parameter learning with reduced optimizer-state memory. | Can memory savings unlock larger local batch/model regimes? |
| Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients | 1 | https://arxiv.org/abs/2407.08296 | Combines gradient projection with quantization; claims pretraining a 7B model on 16GB GPU memory. | Is the result reproducible and stable enough for our stack? |
| 8-bit Optimizers via Block-wise Quantization | 1 | https://arxiv.org/abs/2110.02861 | Reduces optimizer-state memory while preserving performance. | Does it compose with GaLore/Q-GaLore or mostly save the same memory? |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2 | https://arxiv.org/abs/2305.14314 | Commodity fine-tuning breakthrough, but not pretraining. | Which quantized-training lessons transfer to from-scratch or continued pretraining? |
| FP8-LM: Training FP8 Large Language Models | 2 | https://arxiv.org/abs/2310.18313 | Low-precision training for LLMs. | Which parts work on consumer and DGX Spark-class hardware? |

## Conditional Compute And Sparse Capacity

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| Switch Transformers | 2 | https://arxiv.org/abs/2101.03961 | Sparse expert activation gives more capacity per token FLOP. | Does MoE help commodity training, or does routing/communication overhead dominate? |
| GLaM: Efficient Scaling of Language Models with Mixture-of-Experts | 2 | https://arxiv.org/abs/2112.06905 | Strong sparse-activation efficiency evidence at large scale. | Are there small-scale MoE regimes that actually help? |
| DeepSeekMoE: Towards Ultimate Expert Specialization | 2 | https://arxiv.org/abs/2401.06066 | Modern expert-specialization architecture. | Can expert specialization reduce total training tokens? |
| Mixtral of Experts | 2 | https://arxiv.org/abs/2401.04088 | Widely used sparse MoE reference model. | What does its architecture imply for small local MoE experiments? |

## Architecture Alternatives And Edge Cases

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2 | https://arxiv.org/abs/2312.00752 | Alternative to attention with different memory/sequence scaling. | Is training cheaper for our target sizes, or just long-context inference? |
| RWKV: Reinventing RNNs for the Transformer Era | 2 | https://arxiv.org/abs/2305.13048 | RNN-like training/inference characteristics with Transformer-like behavior. | Could recurrence plus simple kernels help commodity training? |
| Retentive Network: A Successor to Transformer for Large Language Models | 2 | https://arxiv.org/abs/2307.08621 | Hybrid retention mechanism with efficient inference claims. | Does retention improve training efficiency or mainly deployment? |

## Methodology And Baseline Calibration

| Source | Tier | Link | Why It Matters | First Question |
| --- | --- | --- | --- | --- |
| Training Compute-Optimal Large Language Models | 1 | https://arxiv.org/abs/2203.15556 | Chinchilla-style compute-optimal baseline for dense training. | What baseline do we compare against for small local models? |
| Measuring the Algorithmic Efficiency of Neural Networks | 2 | https://arxiv.org/abs/2005.04305 | Framework for algorithmic efficiency progress over time. | How do we avoid fake `10x` accounting? |

## First Paper Notes To Write

1. Lottery Ticket Hypothesis
2. Drawing Early-Bird Tickets
3. Dynamic Sparse Training with Structured Sparsity
4. DataComp-LM
5. DoReMi
6. Cramming
7. Parcae
8. Reasoning with Latent Thoughts
9. GaLore
10. Q-GaLore

## Known Open Questions

- Which source was the "Meta study" adjacent to lottery-ticket thinking that originally came to mind? Candidate matches include LIMA for high-quality small data, LLaMA for efficient open pretraining choices, or a Meta pruning/sparsity paper not yet identified.
- Which methods are genuinely multiplicative versus saving the same memory, FLOPs, or tokens under different labels?
- Which gains survive when evaluated on reasoning tasks rather than perplexity or narrow benchmarks?
- Which candidate methods can be tested on RTX 4070 before spending DGX Spark time?
