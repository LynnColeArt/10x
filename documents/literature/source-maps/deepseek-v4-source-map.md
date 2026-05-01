# DeepSeek V4 Source Map

DeepSeek V4 is useful here as a frontier-lab systems case study, not as a directly reproducible commodity recipe.

The local raw video transcript is `deepseek.txt` in the project root. It is treated as a secondary explainer artifact and is not committed by default. Claims below should be grounded in the official report or official API release notes before they enter the evidence matrix.

## Primary Sources

- Official API release note: https://api-docs.deepseek.com/news/news260424
- Official technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- Official transparency entry: https://www.deepseek.com/en/transparency/
- Official model collection: https://huggingface.co/collections/deepseek-ai/deepseek-v4
- Official inference implementation: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference
- MegaMoE / DeepGEMM implementation reference: https://github.com/deepseek-ai/DeepGEMM/pull/304

## Secondary Source

- Video transcript supplied locally for: `The insane engineering of Deepseek V4`, AI Search, uploaded 2026-05-01.

Use the transcript to identify mechanisms and plain-language interpretations. Do not use it as the authority for quantitative claims.

## Transcript Claim Triage

| Transcript Theme | Official-Source Status | Research Handling |
| --- | --- | --- |
| V4-Pro has `1.6T` total parameters and `49B` activated parameters | Confirmed by API note and report | Use as model-scale context, not as commodity target |
| V4-Flash has `284B` total parameters and `13B` activated parameters | Confirmed by API note and report | Treat Flash as the more relevant architecture clue for efficient active-parameter design |
| Both models support `1M` context | Confirmed by API note and report | Useful for long-context efficiency, not direct from-scratch training efficiency |
| Hybrid attention combines CSA, HCA, and sliding-window attention | Confirmed in report; transcript simplifies the details | Convert into toy CSA/HCA experiment candidates |
| V4-Pro uses about `27%` of V3.2 single-token inference FLOPs and `10%` KV cache at `1M` context | Confirmed in report | Count as long-context inference/KV efficiency only |
| V4-Flash uses about `10%` FLOPs and `7%` KV cache versus V3.2 at `1M` context | Confirmed in report | Strong inference-efficiency clue; does not imply `10x` cheaper training |
| mHC prevents signal explosion | Directionally confirmed; official mechanism is a doubly stochastic residual mapping with spectral norm bounded by 1 | Test stability effects in deep or looped toy models |
| mHC overhead is about `6.7%` | Confirmed in report for their optimized overlapped pipeline stage | Do not assume commodity kernels get this overhead |
| Muon replaces AdamW | Partially confirmed; Muon is used for most modules, AdamW remains for embeddings, prediction head, RMSNorm weights, and selected mHC parameters | Needs standalone optimizer note and small-model comparison |
| Training uses curriculum from shorter to longer sequence lengths | Confirmed: training starts at `4K` and extends to `16K`, `64K`, and `1M` | Candidate for commodity long-context curriculum experiments |
| Anticipatory routing stabilizes training | Confirmed with nuance: it computes routing indices using historical parameters and activates dynamically around loss spikes | Strong "accidental finding" candidate for tiny MoE stability experiments |
| Benchmarks rival top closed models | Claimed by official docs/report, but independent replication still needed | Keep outside our efficiency ledger unless capability preservation is central |

## Mechanisms To Extract

- **Hybrid attention:** CSA compresses KV blocks and uses sparse top-k selection; HCA applies heavier compression with dense attention over compressed entries; both include a local sliding-window branch.
- **Precision placement:** DeepSeek uses BF16 for RoPE dimensions, FP8 for remaining KV dimensions, FP4 for indexer attention, and FP4 QAT for routed expert deployment.
- **Residual stability:** mHC constrains residual mapping to the Birkhoff polytope, creating a non-expansive mapping that is designed to reduce deep-stack instability.
- **Optimizer shift:** Muon is used for most non-exempt modules with hybrid Newton-Schulz iterations, while AdamW is retained where Muon is less appropriate.
- **Training curriculum:** The training run starts with shorter contexts, later extends sequence length, and only introduces sparse attention after earlier dense-attention warmup.
- **Routing stabilization:** Anticipatory routing and SwiGLU clamping were introduced after loss spikes and outliers in MoE layers became a practical training failure mode.
- **Systems layer:** MegaMoE, TileLang, deterministic kernels, hybrid ZeRO for Muon, and contextual parallelism are part of the result, not incidental implementation decoration.

## Non-Obvious Clues

- A large share of the "gain" comes from refusing to treat long context as dense attention. This is an attention-allocation problem, not just a faster-kernel problem.
- Stability mechanisms may be training-efficiency mechanisms if they avoid failed runs, rollbacks, and conservative hyperparameter schedules.
- DeepSeek did not rely on one magic trick. The report is a stack of partial gains: attention compression, sparse retrieval, precision placement, MoE active-parameter control, optimizer choice, curriculum, routing stabilization, and kernel work.
- Some gains are only visible because the model is large enough for communication and KV-cache bottlenecks to dominate. We must re-test which mechanisms survive below frontier scale.

## Follow-Up Notes To Write

- Muon optimizer standalone note.
- Manifold-Constrained Hyper-Connections note.
- DeepSeek Sparse Attention / DSA note.
- Tiny MoE routing-stability experiment card.
- CSA/HCA toy long-context attention experiment card.

