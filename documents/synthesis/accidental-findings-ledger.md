# Accidental Findings Ledger

This ledger tracks efficiency-relevant clues that emerged as side effects of work with another stated goal.

The point is not to overclaim. The point is to notice when a paper was trying to solve one problem and accidentally exposed a different kind of waste in current training recipes.

| Source | Original Goal | Side-Effect Signal | Why It Matters | Follow-Up |
| --- | --- | --- | --- | --- |
| Lottery Ticket Hypothesis | Understand why pruning and initialization matter | Dense training may be discovering a much smaller trainable circuit | Suggests circuit discovery and circuit fitting might be separable | Ask whether tickets can be identified early in tiny LMs |
| Early-Bird Tickets | Reduce pruning/search cost in deep networks | Useful masks can stabilize far earlier than full convergence | Turns lottery-ticket thinking from post-hoc compression into possible training savings | Run mask-stability probes on tiny Transformers |
| Structured RigL | Make dynamic sparse training hardware-realistic | Sparse topology must be shaped for kernels, not just parameter count | Prevents fake FLOP accounting and points toward hardware-legible tickets | Test structured sparse Transformer projections |
| DataComp-LM | Benchmark language-model data curation | Dataset construction can rival or exceed model-recipe changes at fixed compute | Data quality may be one of the largest realistic multipliers | Build miniature curation loop before architecture experiments |
| Parcae | Stabilize looped language models and study scaling | Parameter reuse can recover quality of larger static-depth models | Recurrent depth may trade parameter memory for adaptive compute | Measure quality-per-wall-clock, not only quality-per-parameter |
| DoReMi | Optimize data-mixture proportions without downstream labels | A `280M` proxy can improve training of an `8B` target model | Cheap proxy models may guide expensive training decisions across data, masks, and curricula | Build proxy-guided miniature data-mixture experiment |
| Cramming | Train a language model from scratch in one day on one GPU | Under fixed commodity budgets, many fashionable large-scale tricks do not help while pipeline/data details matter | Keeps the project honest about what works outside frontier-lab scale | Use as RTX 4070 harness discipline |
| Reasoning with Latent Thoughts | Explain looped Transformers as latent reasoning systems | Effective reasoning depth may not require unique parameters at every layer | Suggests memory-constrained models can trade loops for depth-like computation | Compare looped versus static-depth tiny models at matched wall-clock |
| GaLore | Reduce optimizer-state memory | Full-parameter pretraining becomes feasible in prosumer GPU memory classes | Training accessibility can improve even before FLOPs improve | Reproduce small LLaMA-style run and separate memory from wall-clock |
| Q-GaLore | Reduce GaLore memory and SVD overhead | Gradient subspaces stabilize at different rates by layer | Adaptive optimizer effort may be a general training-efficiency seam | Measure whether lazy subspace updates preserve loss while improving speed |
| DeepSeek-V4 | Make `1M` context practical in a frontier MoE model | Loss spikes, routing outliers, dense long-context attention, and non-deterministic systems behavior appear as hidden compute taxes | Training efficiency may come from preventing wasted runs and only spending attention where needed, not just reducing nominal parameter count | Test Muon, CSA/HCA, mHC, sequence-length curriculum, and anticipatory routing as separate local mechanisms |
