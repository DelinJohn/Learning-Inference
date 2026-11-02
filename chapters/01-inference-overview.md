# Chapter 01 — Inference Overview: Prefill vs Decode, Compute vs Memory

> The big picture. Read this first to understand how the rest of the chapters fit together.

## Central idea

An LLM request normally has two phases:

1. **Prefill:** process all prompt tokens, usually in parallel, and construct the KV cache.
2. **Decode:** generate new tokens autoregressively, usually one token per sequence per forward pass, while reading and extending the KV cache.

The usual performance characterization is:

| Phase or operation | Typical bottleneck |
|---|---|
| Prefill, including KV-cache construction | Compute throughput |
| Token-by-token decode at a small batch size | Memory bandwidth |
| Storing the KV cache | Memory capacity |
| Reading the KV cache during decode | Memory bandwidth |

The important clarification is that **the KV cache itself is not inherently compute-bound**. The statement "LLM prefill (KV-cache construction) is compute-bound" describes the complete prefill workload during which the cache is created. Reading the completed cache during decode has very different performance characteristics.

## What "compute-bound" and "memory-bound" mean

A workload is **compute-bound** when execution time is mainly limited by how quickly the processor can perform arithmetic.

A workload is **memory-bandwidth-bound** when execution time is mainly limited by how quickly data can be transferred from memory to the compute units. More available FLOPs will not help much if the compute units are waiting for data.

The useful quantity is **arithmetic intensity**:

$$
\text{Arithmetic intensity}
=
\frac{\text{number of arithmetic operations}}{\text{bytes transferred from memory}}
$$

A high arithmetic intensity tends to make an operation compute-bound. A low arithmetic intensity tends to make it memory-bound. The exact boundary depends on the GPU's ratio of peak compute throughput to memory bandwidth; this is the basic idea behind the roofline model.

## Prefill and decode side by side

| Property | Prefill | Decode |
|---|---|---|
| Tokens processed per sequence per pass | Entire prompt | Usually one new token |
| Main linear algebra | Matrix–matrix | Matrix–vector or small matrix–matrix |
| Weight reuse within a pass | High | Low at small batch sizes |
| KV behavior | Computes and writes cache | Reads old cache and appends one entry |
| Typical arithmetic intensity | High | Low |
| Typical bottleneck | Compute | Memory bandwidth |
| User-facing metric | Time to first token (TTFT) | Time per output token (TPOT) |

## Important qualifications

"Prefill is compute-bound and decode is memory-bound" is a useful default model, not a universal law.

The actual bottleneck depends on:

- model architecture and model size;
- prompt and context length;
- batch size and continuous batching behavior;
- data type and quantization;
- GPU compute-to-bandwidth ratio;
- attention implementation and kernel efficiency;
- tensor, pipeline, expert, or context parallelism;
- communication between GPUs;
- Mixture-of-Experts routing;
- speculative or multi-token decoding.

Short prefills or poorly utilized kernels can be bandwidth-bound. Large decode batches can provide enough weight reuse to become more compute-bound. Long-context decoding can be dominated specifically by KV-cache bandwidth, whereas shorter-context decoding is often dominated by loading model weights.

## Compact mental model

Use this question:

> After loading a value from memory, how many times can the GPU use it for arithmetic?

- **Prefill:** loaded weights and attention data are reused across many prompt tokens, so there is substantial computation per byte.
- **Small-batch decode:** loaded weights and cached keys/values are used for only one or a few new tokens, so there is little computation per byte.

In one sentence:

> Prefill spends heavily on math to construct the KV cache; decode saves that recomputation but must continually stream weights and the existing cache from memory.

---

**Continue to:** [02-prefill-decode-mechanics](./02-prefill-decode-mechanics.md) for the detailed mechanics of each phase.
