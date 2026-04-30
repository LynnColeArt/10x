# Replication Chatter: DataComp-LM And DoReMi

## Scope

Sources reviewed:

- DCLM repo issue list and selected issues: https://github.com/mlfoundations/dclm/issues
- DoReMi repo issue list and selected issues: https://github.com/sangmichaelxie/doremi/issues
- DataComp-LM repo: https://github.com/mlfoundations/dclm
- DoReMi repo: https://github.com/sangmichaelxie/doremi

## Main Signal

Data efficiency still looks like the strongest near-term `10x` lane, but chatter shows that it is not a lightweight trick. The hidden costs are pipeline infrastructure, exact filtering parameters, dataset access, tokenizer choice, evaluation comparability, and data-shuffling/tokenization memory.

## DataComp-LM Landmines

### Dedup And Bloom Filters Can Be Huge

DCLM issue #99 reports a dedup/BFF bloom filter consuming roughly `90%` of RAM on a server with `836GB` available while processing about `12TB` of text: https://github.com/mlfoundations/dclm/issues/99

Implication:

- Full-scale DCLM-style curation is not commodity by default.
- For our project, DCLM should become a miniature curation recipe first, not a full-scale reproduction.

### Dataset Access And Mirrors Matter

DCLM issue #103 reports AWS S3 access denial when trying to copy DCLM shards; maintainers point to Hugging Face mirrors as an alternate route: https://github.com/mlfoundations/dclm/issues/103

Implication:

- Data availability is part of reproducibility.
- Any experiment card must pin exact dataset location, version, and fallback mirror.

### Tokenization/Shuffling Can Be Memory-Bound

DCLM issue #88 reports local tokenization memory problems for the 400M-1x data. A maintainer says the memory-heavy behavior is by design because shuffling is needed for training and suggests more space or a Rust tokenization/shuffling path: https://github.com/mlfoundations/dclm/issues/88

Implication:

- Preprocessing must be included in the cost ledger.
- If tokenization/shuffling becomes the bottleneck, a "training efficiency" win may shift cost into data preparation.

### Reproducing Small Settings Requires Exact Parameters

DCLM issue #120 reports a replication attempt for the 400M-1x experiment with retention differences after dedup and fastText filtering, asking for exact BFF parameters and token-count accounting: https://github.com/mlfoundations/dclm/issues/120

Implication:

- Small curation experiments are sensitive to retention thresholds and exact processing parameters.
- We should write down each filtering stage's input/output counts.

## DoReMi Landmines

### Reproduction Is Sensitive To Tokenizer, Seeds, FlashAttention, And Eval Details

DoReMi issue #20 is a long reproduction discussion. Users report mismatched 120M results and unstable SQuAD behavior. The maintainer notes FlashAttention version differences, CUDA version, seed sensitivity, and that SQuAD few-shot eval is unstable at 120M scale: https://github.com/sangmichaelxie/doremi/issues/20

Implication:

- DoReMi's proxy-data-mixture idea is still valuable, but our experiment must use fixed seeds, fixed eval examples, and explicit per-domain validation.

### Data Loader Behavior Can Create Large Speed Drops

DoReMi issue #24 reports a sudden speed drop from about 2 iterations/second to 6 seconds/iteration during training on 4 A100 nodes. The maintainer suspects on-the-fly caching of shuffle indices: https://github.com/sangmichaelxie/doremi/issues/24

Implication:

- Data pipeline behavior can dominate observed training speed.
- Our proxy-mixture experiments need to log dataloader time separately.

### Domain Weights May Transfer, But Tokenizer Is A Major Variable

DoReMi issue #23 asks whether released domain weights can be applied to different training configs. The maintainer says the tokenizer appears to be the biggest factor because it changes the data itself, with some expected degradation when hyperparameters differ: https://github.com/sangmichaelxie/doremi/issues/23

Implication:

- We should not treat learned domain weights as universal.
- Any local DoReMi experiment should learn weights with the same tokenizer used in the target run.

## Positive Signals

- DoReMi maintainer responses preserve the main result after controlling seed and excluding unstable SQuAD, while acknowledging fragility.
- DCLM maintainers provide alternative dataset access paths and concrete implementation guidance, which helps reproducibility.
- Both lanes expose a reusable pattern: cheap proxy or preprocessing work can save expensive model-training steps, but only if preprocessing is counted.

## Experiment Impact

For DataComp/DoReMi experiment cards, require:

- Preprocessing wall-clock.
- Peak RAM and disk for filtering/tokenization.
- Dataset retention by stage.
- Tokenizer hash/version.
- Fixed eval examples and seeds.
- Per-domain validation, not only weighted aggregate validation.
- Dataloader time versus model step time.

## Provisional Verdict

Still the strongest near-term lane, but the first test should be a miniature curation/proxy-mixture loop with strict cost accounting rather than a full DCLM reproduction.
