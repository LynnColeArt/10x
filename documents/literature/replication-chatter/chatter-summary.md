# Chatter Summary

## First-Pass Takeaway

The chatter pass strengthens the project, but it also makes the accounting problem sharper. Most promising methods do not simply reduce training cost. They shift cost among memory, preprocessing, data movement, projection overhead, token budget, and evaluation stability.

## High-Value Landmines

| Lane | Landmine | Why It Matters |
| --- | --- | --- |
| Data curation | DCLM-style dedup/tokenization can require huge RAM/disk and exact filtering parameters | Preprocessing cost must be counted in the `10x` ledger |
| Data mixtures | DoReMi reproduction is sensitive to tokenizer, seeds, FlashAttention, and unstable tasks | Proxy-data gains need controlled eval and fixed seeds |
| Memory optimizers | GaLore/Q-GaLore can fit larger models but SVD/projection overhead may dominate time | Memory accessibility is not the same as wall-clock efficiency |
| Commodity harness | Cramming results depend on data wraparound, compile settings, package versions, and microbatch saturation | Our RTX 4070 baseline must pin environment and report GPU utilization |
| Sparse training | SRigL is constant fan-in, not arbitrary hardware `N:M`; attention internals may remain dense | Sparse results need real kernel-backed wall-clock proof |
| Recurrent depth | Parcae needs explicit GPT baselines and token-budget caution | Recurrent depth is parameter efficiency until wall-clock proof exists |

## Most Actionable Next Experiments

1. **Mini DataComp/DoReMi Loop:** small corpus, fixed tokenizer, staged filtering, proxy mixture weights, fixed eval seeds, full preprocessing cost ledger.
2. **GaLore/Q-GaLore Smoke Test:** 60M/130M model, AdamW versus 8-bit AdamW versus GaLore/Q-GaLore, with projection time and checkpoint/resume tests.
3. **Cramming-Derived Baseline Harness:** fixed-token and fixed-wall-clock autoregressive baseline, with environment lock and GPU utilization.
4. **Sparse Mask Probe:** tiny Transformer mask-stability plus structured projection sparsity, reporting actual tokens/sec.
5. **Recurrent-Depth Probe:** Parcae versus GPT-style baseline at matched parameter, matched FLOP, and matched wall-clock budgets.

## Research Posture Update

The project should now distinguish:

- **Training compute efficiency:** fewer FLOPs or fewer optimizer steps to same quality.
- **Wall-clock efficiency:** less elapsed time to same quality on target hardware.
- **Memory accessibility:** enables training that otherwise would not fit.
- **Data efficiency:** fewer or better tokens to same quality.
- **Pipeline efficiency:** lower preprocessing/data movement overhead.
- **Capability efficiency:** same reasoning/product utility at lower training cost.

No candidate gets full `10x` credit unless it improves the relevant axis without hiding the cost in another axis.
