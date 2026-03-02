<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4090 (24GB)
- Python: 3.12.12; torch: 2.9.1+cu128; triton: 3.5.1; transformers: 4.57.6; flash_attn: 2.8.3
- num-speculative-tokens=5
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Model | Speculative Model | Total Requets | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|------------|------------|-----|----------|----------|----------------------|
| vLLM           | Qwen3-0.6B | None       | 256 | 133966   | 20.02    | 6691.74              |
| Nano-vLLM      | Qwen3-0.6B | None       | 256 | 133966   | 22.27    | 6015.05              |
| vLLM           | Qwen3-8B   | None       | 256 | 133966   | 124.19   | 1078.68              |
| Nano-vLLM      | Qwen3-8B   | None       | 256 | 133966   | 152.57   | 878.08               |
| Nano-vLLM      | Qwen3-8B   | Qwen3-0.6B | 256 | 133966   | 265.42   | 504.73               |
| Nano-vLLM      | Qwen3-8B   | None       | 64  | 38443    | 45.76    | 840.08               |
| Nano-vLLM      | Qwen3-8B   | Qwen3-0.6B | 64  | 38443    | 77.42    | 496.57               |
| Nano-vLLM      | Qwen3-8B   | None       | 8   | 3739     | 15.58    | 240.06               |
| Nano-vLLM      | Qwen3-8B   | Qwen3-0.6B | 8   | 3739     | 12.00    | 311.64               |
| Nano-vLLM      | Qwen3-8B   | None       | 1   | 724      | 12.73    | 56.87                |
| Nano-vLLM      | Qwen3-8B   | Qwen3-0.6B | 1   | 724      | 6.90     | 104.97               |

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date) -->
