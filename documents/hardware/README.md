# Hardware

This folder tracks the actual hardware constraints for experimental validation.

## Known Hardware

### DGX Spark Founders Edition

- Role: primary experimental training target.
- Details: to be profiled before main experiments.
- Open questions: memory capacity, supported low-precision modes, storage bandwidth, software stack, thermal/power constraints.

### NVIDIA RTX 4070

- Role: local smoke tests, small ablations, harness debugging.
- Details: to be profiled before baseline work.
- Open questions: available VRAM, CUDA version, supported precision modes, practical batch sizes.

## Profiling Tasks

- Capture GPU model, VRAM, driver, CUDA, and PyTorch versions.
- Measure baseline memory use for a tiny Transformer training step.
- Measure storage throughput for dataset streaming.
- Decide which experiments can run locally versus DGX Spark only.
