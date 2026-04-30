# Replication Chatter: Recurrent Depth

## Scope

Sources reviewed:

- Parcae repo issue list and selected issues: https://github.com/sandyresearch/parcae/issues
- Parcae repo metadata: https://github.com/sandyresearch/parcae

## Main Signal

Parcae is new and active, so there is little mature replication chatter. The early public issues are still valuable because they point to the exact comparison risk for recurrent-depth work: fixed-depth GPT baselines, token budget, and stochastic versus fixed recurrence configuration.

## Landmines

### Fair GPT Baselines Need Explicit Scripts

Parcae issue #3 asks for a verified standard Transformer/GPT pretraining script to compare against Parcae RDM-style training. A maintainer points to `gpt-small-140m.yaml` and explains settings for fixed recurrence: https://github.com/sandyresearch/parcae/issues/3

Implication:

- Our recurrent-depth experiment card must include matched GPT baseline scripts from the start.
- We should not rely on README defaults alone.

### Claims Are Already Getting Token-Budget Pushback

Parcae issue #4 argues that some experiments trained on small token budgets or few optimization steps may not justify broad pretraining conclusions: https://github.com/sandyresearch/parcae/issues/4

Implication:

- The recurrent-depth lane must be careful about extrapolating from small-token experiments.
- We need to label early runs as mechanism probes, not pretraining proof.

### Documentation Is Still Moving

Parcae issue #1 reports a documented launch config missing a training budget; the issue was closed after maintainers addressed it: https://github.com/sandyresearch/parcae/issues/1

Implication:

- Pin commits for any Parcae reproduction.
- Expect README/config drift while the project is fresh.

## Positive Signals

- Maintainers are responsive and the repo is active.
- The project already exposes both Parcae and GPT-style configs, which is useful for our matched-baseline requirement.
- The discussion around fixed versus stochastic recurrence is exactly the kind of detail our experiment cards need.

## Experiment Impact

For recurrent-depth experiment cards, require:

- Matched parameter baseline.
- Matched training FLOP baseline.
- Matched wall-clock baseline.
- Fixed recurrence and stochastic recurrence variants.
- Token budget sweep.
- Separate reporting for parameter efficiency, training efficiency, and inference/test-time compute.

## Provisional Verdict

Keep recurrent depth as a high-interest mechanism lane, but do not credit it as training efficiency until matched-wall-clock and matched-token experiments survive.
