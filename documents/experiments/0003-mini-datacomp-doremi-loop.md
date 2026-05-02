# Experiment Card: Mini DataComp/DoReMi Loop

## Question

Can a cheap proxy model choose or reweight training data so a target small LM reaches the same quality with fewer tokens or less wall-clock time?

## Hypothesis

If data quality and mixture selection are a major hidden multiplier, a proxy-guided mixture should beat a uniform or hand-chosen mixture on validation loss and reasoning probes after all preprocessing cost is counted.

## Baseline

- Model: `probe-60m` dense decoder-only causal LM.
- Dataset: same source corpus manifest as the baseline, sampled with fixed uniform or hand-set mixture weights.
- Tokenizer: same tokenizer as baseline.
- Training budget: fixed target tokens and fixed wall-clock.
- Hardware: RTX 4070 for proxy and target probe runs; DGX Spark only after the method is stable.
- Metrics: proxy cost, preprocessing cost, target validation loss, time-to-loss, tokens-to-loss, eval probes, source-slice coverage, and data-loader throughput.

## Intervention

Use a small proxy model to estimate useful mixture weights before training the target model.

Initial loop:

- Train or partially train a `smoke-15m` proxy on candidate data slices.
- Score slices by held-out loss improvement, gradient signal, or DoReMi-like excess-loss weighting.
- Produce a fixed mixture for the `probe-60m` target.
- Train the target on the proxy-selected mixture.
- Compare against the baseline mixture under identical token and wall-clock budgets.

## Expected Gain

- FLOPs: Target training FLOPs should fall only if fewer target tokens or steps reach the same quality.
- Wall-clock: Target time-to-loss should improve, but proxy and preprocessing time must be reported.
- Memory: No direct gain expected.
- Data: Target `1.2x-2x` fewer target tokens to equal validation/eval quality before treating it as promising.
- Quality: Equal or better validation loss and stronger deterministic reasoning probes.

## Evaluation

- Language modeling: overall validation loss plus per-slice validation loss.
- Reasoning: arithmetic, small math/reasoning prompts, and structured-output probes.
- Robustness: fixed seeds, stable mixture weights across at least two proxy seeds if promising.
- Regression checks: no collapse into narrow-domain behavior, no major degradation on general-language validation.

## Hardware Target

- RTX 4070 smoke test: build corpus manifest, preprocessing ledger, and proxy scoring loop.
- RTX 4070 probe run: `probe-60m` target with baseline mixture versus proxy mixture.
- DGX Spark main run: larger target only if proxy cost is clearly smaller than target-training savings.

## Failure Criteria

Downgrade this path if:

- Preprocessing plus proxy cost exceeds saved target-training time.
- Mixture weights are unstable across seeds.
- Gains appear only on the validation set used to choose the mixture.
- The target improves perplexity but regresses reasoning or robustness probes.
- Data loading becomes the bottleneck and hides the training-side result.

## Notes

This is likely one of the highest-upside lanes because it attacks wasted tokens directly. It is also one of the easiest places to lie to ourselves, so the preprocessing ledger is part of the experiment, not paperwork.
