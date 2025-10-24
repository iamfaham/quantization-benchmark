# quantization-benchmark

A small repository for benchmarking model quantization workflows and measuring the accuracy / performance tradeoffs introduced by different quantization approaches.

---

Repository snapshot
- Repository: iamfaham/quantization-benchmark
- Current files (at this commit):
  - README.md — https://github.com/iamfaham/quantization-benchmark/blob/main/README.md
- Repo commit OID: 498b14339ae640dc20c65d41cd1877af38ddde52
- README blob SHA: 874b16522649aade82f4d0e1644c40a3e30c099c

> Note: At the moment this repository contains only this README file. The sections below describe the intended purpose, suggested layout, and contribution guidelines to make the repository useful for benchmarking quantization approaches.

---

What this repo is for
- Provide reproducible benchmarks comparing full-precision and quantized models (e.g., FP32 vs INT8).
- Track accuracy, latency, memory, and model size after applying different quantization strategies and toolchains.
- Store scripts and example configurations to run end-to-end quantization experiments.

Suggested repository layout
- data/ — datasets and dataset-preparation scripts (or instructions to download them).
- models/ — model definitions and pretrained weights (or links to them).
- scripts/ — command-line scripts for running training, evaluation, and quantization pipelines.
- benchmarks/ — benchmark harnesses, runner scripts, and experiment configurations.
- results/ — serialized benchmark outputs, CSVs, and plots.
- notebooks/ — exploratory notebooks for visualizing and inspecting results.
- docs/ — documentation for experiments and methods used.

Usage (example commands to include when scripts are added)
- Clone the repo
  git clone https://github.com/iamfaham/quantization-benchmark.git
- Typical workflow (to be implemented in scripts/)
  1. Prepare dataset: scripts/prepare_data.sh
  2. Evaluate FP32 baseline: scripts/eval.sh --model models/resnet50_fp32.pth --dataset data/imagenet
  3. Quantize model: scripts/quantize.sh --model models/resnet50_fp32.pth --method post_training_static
  4. Run quantized evaluation: scripts/eval.sh --model models/resnet50_int8.pth --dataset data/imagenet
  5. Collect metrics: benchmarks/collect_results.py

Recommended benchmark metrics
- Accuracy (top-1, top-5)
- Inference latency (p50 / p95)
- Throughput (images/sec)
- Model size on disk
- Peak memory usage
- Quantization method & configuration (e.g., per-channel vs per-tensor, calibration dataset size)

How to contribute
- Add code, benchmarks, or datasets under appropriately named directories.
- Create small, focused pull requests with:
  - A clear description of what was added/changed
  - A short README or note describing how to run the added benchmark
  - Example commands to reproduce results
- Include tests where reasonable (e.g., small unit tests for scripts).

License and attribution
- No license file is present currently. Add a LICENSE file if you want to specify reuse terms.

Contact
- Maintainer: iamfaham
- For questions about experiments or contributions, open an issue or submit a pull request with details.

---

If you want, this README can be adjusted to list any existing scripts, models, or datasets once they're added to the repository.