# Quantization Benchmark Playground

> A practical notebook for benchmarking **FP16 vs INT4** quantized language model inference using `bitsandbytes` and the Hugging Face `transformers` stack.

---

### Overview

This notebook explores how **post-training quantization** affects the performance, accuracy, and efficiency of decoder-only transformer models.

It measures:

| Metric         | Description                                          |
| -------------- | ---------------------------------------------------- |
| **Perplexity** | Evaluates model accuracy on the WikiText-2 test set  |
| **Latency**    | Generation speed (time to produce new tokens)        |
| **GPU Memory** | Peak CUDA memory during inference                    |

The benchmark demonstrates how lower precision (4-bit INT4) compares to standard FP16 precision in real inference scenarios.

---

### Experiment Setup

**Model:** [facebook/opt-66m](https://huggingface.co/facebook/opt-66m)
**Quantization Library:** [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
**Dataset:** WikiText-2 (test split)
**Hardware:** NVIDIA T4 GPU (Colab tested)

Each test:

1. Loads a pretrained OPT model (FP16 and INT4 versions).
2. Generates text sequences (default: 200 new tokens).
3. Measures latency, memory, and perplexity.
4. Compares results between FP16 and quantized versions.

---

### Theoretical Background

**Quantization** compresses model weights by lowering their numeric precision.

* **FP16:** 16-bit floating-point (≈2 bytes/weight)
* **INT4:** 4-bit integer (≈0.5 bytes/weight)

This can shrink model size by ~4×, but adds unpacking overhead during computation.

In transformer models:

* Weights are quantized.
* KV-caches (used during generation) remain FP16 — so runtime memory doesn’t always drop.

---

### Results Summary

| Metric          | FP16    | INT4    | Observation                           |
| --------------- | ------- | ------- | ------------------------------------- |
| **Perplexity**  | ~70.9   | ~74.7   | ✅ Minimal accuracy loss (<6%)         |
| **Latency**     | ~1.9 s  | ~5.9 s  | ⚠️ INT4 slower (small model overhead) |
| **Peak Memory** | ~9.2 GB | ~9.2 GB | ⚠️ KV-cache dominates memory          |

> **Key takeaway:**
> Quantization preserves accuracy and reduces weight size but may not improve latency for smaller models.
> The benefits become significant with **larger architectures** (≥350 M parameters) and **longer sequences**.

---

### Why INT4 Appears Slower on Small Models

1. **Kernel Overhead** — bitsandbytes launches custom CUDA kernels, which dominate small compute graphs.
2. **KV-Cache Unquantized** — most runtime memory is still FP16.
3. **Short Sequences** — quantization overhead isn’t amortized over long runs.
4. **Small Model Size** — compute load too small for parallelization benefits.

---

### Lessons Learned

* Quantization can maintain model quality with negligible accuracy drop.
* True efficiency gains emerge with **larger models** and **longer token generation**.
* Always distinguish between **model-weight memory** and **runtime memory**.
* Benchmarks must include both **accuracy (PPL)** and **latency** to assess real trade-offs.

---

### How to Run

1. Clone this repository:
   code
   git clone [https://github.com/iamfaham/quantization-benchmark.git](https://github.com/iamfaham/quantization-benchmark.git)
   cd quantization-benchmark
   code

2. Launch the notebook:
   code
   jupyter notebook Quantization_Benchmark.ipynb
   code
---

### Dependencies

```
pip install torch transformers datasets bitsandbytes accelerate
```

> Tested with:
>
> * `transformers >= 4.44`
> * `bitsandbytes >= 0.43`
> * `torch >= 2.3`
> * Python 3.10+

---

### Future Improvements

* Add **GPTQ** and **AWQ** quantization backends.
* Profile with **NVIDIA TensorRT-LLM** for low-level kernel comparison.
* Test on larger models (OPT-350M, Mistral-7B, LLaMA-2-7B).
* Include **weight-only memory profiling** and **KV-cache quantization**.
