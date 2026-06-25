# nano-vLLM Deep Dive

An 11-chapter study series on LLM inference engineering — from tokens to PagedAttention — built around the [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) codebase.

## What This Is

A beginner-friendly yet technically rigorous guide to how large language model inference actually works. Each chapter is a self-contained HTML page with:

- Real-world analogies before every new concept
- Interactive visualizations and simulators
- Annotated code from the actual nano-vLLM implementation
- A 3-question quiz with detailed feedback

No ML, linear algebra, or CUDA knowledge required. Basic Python reading comfort helps — all code is annotated line by line.

## Chapters

| # | Title | Topic | ~Time |
|---|-------|-------|-------|
| 01 | What Is LLM Inference? | Tokens, autoregressive generation, Q/K/V, HBM, bottlenecks | 22 min |
| 02 | nano-vLLM Architecture | File structure, CPU control plane vs GPU data plane | 18 min |
| 03 | KV Cache | Physical tensor layout, Triton kernel, GQA | 20 min |
| 04 | PagedAttention | Virtual memory for KV cache, block tables, free lists | 22 min |
| 05 | The Scheduler | Continuous batching, state machine, preemption | 20 min |
| 06 | Prefill vs Decode | Compute-bound vs memory-bound, TTFT/TPOT, chunked prefill | 18 min |
| 07 | Prefix Caching | xxhash content-addressable blocks, reference counting | 18 min |
| 08 | Sampling Strategies | Greedy, temperature, top-k, top-p | 16 min |
| 09 | Tensor Parallelism | Column/Row parallel layers, all-reduce, NCCL | 20 min |
| 10 | Optimization Stack | FlashAttention, CUDA Graphs, torch.compile, kernel fusion | 24 min |
| 11 | Benchmarks | Throughput vs latency, nano-vLLM vs vLLM, reading benchmarks honestly | 14 min |

**Total reading time: ~4 hours**

## Running Locally

No build step required. Serve the static files:

```bash
cd nano-vllm-guide
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Who This Is For

- **Software engineers** curious about how LLMs work under the hood
- **ML practitioners** who use LLM APIs and want to understand what happens beneath `generate()`
- **Beginners** who've heard "KV cache" or "PagedAttention" and want a real explanation

## Based On

All code examples come from [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a ~1,200 line reimplementation of vLLM's core algorithms in pure Python and Triton, created by Xingkai Yu (DeepSeek).

## License

Apache 2.0
