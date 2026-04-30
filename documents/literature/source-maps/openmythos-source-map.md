# OpenMythos Source Map

OpenMythos is useful here as a hypothesis aggregate, not as proof of any proprietary Anthropic architecture.

## Handling Rules

- Extract underlying primary sources before making claims.
- Separate testable recurrent-depth architecture claims from Claude Mythos speculation.
- Track whether each claim is about training efficiency, inference-time compute, parameter efficiency, or reasoning behavior.
- Prefer papers and runnable code over secondary commentary.

## Initial Claims To Extract

- Recurrent-depth or looped Transformer blocks can reuse parameters across multiple computation steps.
- Adaptive loop count may trade inference compute for reasoning depth.
- Input injection into recurrent blocks may stabilize repeated latent computation.
- Sparse MoE inside a recurrent block may combine parameter reuse with conditional compute.
- MLA/GQA-style attention choices may reduce memory or KV cost.

## Candidate Primary Sources

- Universal Transformers
- Parcae / stable looped language model work
- Reasoning with Latent Thoughts / looped Transformer work
- Training Large Language Models to Reason in a Continuous Latent Space
- Relaxed Recursive Transformers

## Questions For The Dossier

- Does recurrence reduce training compute, or mainly improve parameter efficiency?
- Does recurrence increase inference cost enough to offset training gains?
- Are recurrent-depth gains still present after strong dense baselines?
- Can recurrent blocks be combined with sparse training or early-ticket selection?
