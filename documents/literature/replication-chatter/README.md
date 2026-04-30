# Replication And Chatter

This folder tracks community stress-testing around high-priority papers and repos.

The goal is not to treat chatter as proof. The goal is to find hidden costs, reproduction traps, negative results, ambiguous baselines, and implementation details that papers often compress away.

## Evidence Types

- **Author follow-up:** repo issues, comments, revisions, follow-on papers, or maintainer clarifications.
- **Independent reproduction:** external runs, benchmark attempts, issue reports, or public experiment logs.
- **Implementation landmine:** dependency, hardware, memory, data, or script issue that affects whether a method is practical.
- **Conceptual pushback:** fair-baseline, metric, data-leakage, token-budget, or accounting objection.
- **Practitioner enthusiasm:** useful for prioritization, but low evidentiary weight unless paired with logs or code.

## Current Notes

- [galore-qgalore.md](galore-qgalore.md)
- [datacomp-doremi.md](datacomp-doremi.md)
- [cramming.md](cramming.md)
- [sparse-training.md](sparse-training.md)
- [recurrent-depth.md](recurrent-depth.md)

## Working Rule

Every experiment card should include a `Replication/Chatter Risks` section before we spend serious DGX Spark time.
