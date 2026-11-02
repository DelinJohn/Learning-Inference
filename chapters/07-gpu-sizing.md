# Chapter 07 — Choosing a GPU Systematically

> A repeatable procedure: define the workload, check VRAM, check prefill compute, check decode bandwidth, check software support, then benchmark.

## 1. Define the workload before selecting hardware

Record:

- exact model and architecture;
- weight and KV-cache data types;
- expected and maximum prompt lengths;
- expected output length;
- average and peak concurrent sequences;
- requests per second;
- target time to first token (TTFT);
- target inter-token latency or time per output token (ITL/TPOT);
- target aggregate tokens per second;
- latency percentiles required, such as p95 or p99.

A GPU cannot be selected correctly from parameter count alone. A single-user, 4K-context application and a 100-user, 32K-context service can require very different memory and bandwidth for the same model.

## 2. Read the model configuration

Obtain at least:

- total and, for MoE, active parameter counts;
- number of layers;
- hidden dimension;
- number of query heads;
- number of KV heads;
- head dimension;
- maximum context or sliding-window behavior;
- supported numerical formats and inference kernels.

### How architecture knowledge is used in inference engineering

An inference engineer usually does not invent mechanisms such as KDA or MLA or train a foundation model from scratch. The practical requirement is to recognize the architecture and predict how it will behave for the intended serving workload.

This knowledge is used to:

- estimate weight and KV-cache memory;
- predict whether prompt processing, token generation, bandwidth, or capacity will be the likely bottleneck;
- choose a model for the required context, retrieval accuracy, concurrency, latency, and throughput;
- confirm that the serving engine and GPU have optimized kernels for the architecture;
- benchmark the actual workload rather than relying only on theoretical complexity.

```text
Model-architecture researcher
    → invents or trains mechanisms such as KDA or MLA

Inference engineer
    → understands their consequences, selects a suitable model,
      configures the serving system, measures it, and optimizes deployment
```

Implementing attention mathematics from scratch is useful for learning and may be required for kernel or framework work, but it is not a routine requirement for deploying existing models.

## 3. Check VRAM capacity

Calculate:

$$
M_{\text{weights}}
+ M_{\text{KV at target concurrency/context}}
+ M_{\text{runtime and headroom}}.
$$

Reject any candidate that cannot fit this allocation reliably. If one GPU cannot fit it, consider quantization, reducing context or concurrency, KV-cache quantization, or multi-GPU sharding.

## 4. Check prefill compute

Estimate prompt FLOPs and compare them with achieved—not merely peak—compute throughput. This determines whether the candidate is likely to meet the TTFT target.

## 5. Check decode memory bandwidth

Estimate model-weight bytes per iteration and KV-cache bytes read at representative context lengths. Compare these with effective HBM bandwidth. This is often the most important check for low-batch token-generation speed.

## 6. Check precision and software compatibility

Confirm that the GPU and serving stack efficiently support the desired:

- BF16, FP16, FP8, INT8, or INT4 path;
- attention backend;
- quantization method;
- model architecture;
- tensor/pipeline/expert-parallel implementation.

A format fitting in VRAM is not enough if the GPU lacks fast kernels for it.

## 7. Check multi-GPU communication

Tensor parallelism can approximately divide many weight tensors across GPUs, but it adds collective communication inside transformer layers. Pipeline parallelism splits layers and has different latency and scheduling costs. Data parallelism creates model replicas and improves service capacity, but each replica must still fit its own model allocation.

For tensor parallel degree $G$, a rough starting point is

$$
M_{\text{weights per GPU}}
\approx
\frac{M_{\text{weights}}}{G},
$$

but KV cache, duplicated parameters, buffers, and workspaces are architecture- and engine-dependent. Do not assume total VRAM simply adds together with perfect efficiency. Interconnect bandwidth and topology—such as a fast GPU fabric versus PCIe alone—can determine whether adding GPUs improves or harms latency.

## 8. Benchmark the real serving workload

Analytical calculations produce bounds and eliminate impossible candidates; they do not replace measurement. Benchmark with the real model, quantization, prompt-length distribution, output lengths, concurrency, and scheduler settings. Measure at least:

- peak VRAM use and cache capacity;
- TTFT;
- TPOT/ITL;
- aggregate input and output token throughput;
- p50, p95, and p99 latency;
- queueing, preemption, and out-of-memory behavior;
- performance at both normal and peak load.

## GPU property priority for LLM inference

| GPU property | Why it matters |
|---|---|
| VRAM capacity | Determines whether weights, KV cache, activations, and runtime allocations fit |
| HBM or GPU-memory bandwidth | Often sets small-batch decode speed because weights and KV data must be streamed |
| Tensor Core throughput at the actual data type | Often sets prefill and large-batch matrix-multiplication speed |
| Supported precision and optimized kernels | Determines whether advertised low-precision performance is usable by the model and serving engine |
| GPU interconnect | Matters when weights or computation are sharded across GPUs |
| Power, cooling, and form factor | Determines whether the GPU can physically and electrically run in the target system |
| Measured model performance | Validates TTFT, token latency, throughput, and memory use under the real workload |

This connects directly to the roofline comparison:

- a memory-bound decode workload benefits primarily from more effective memory bandwidth or greater data reuse;
- a compute-bound prefill workload benefits primarily from more effective Tensor Core FLOP/s;
- both phases still require enough VRAM capacity for the target model, context, and concurrency.

Buying a GPU with twice the Tensor Core FLOP/s does not guarantee twice the decode speed if decode is already waiting on HBM. Likewise, a high-bandwidth GPU cannot run a model whose complete allocation does not fit.

## A compact GPU-sizing worksheet

Fill in the following values:

| Input | Symbol or calculation |
|---|---|
| Parameter count | $P$ |
| Active parameters per token for MoE | $P_{\text{active}}$ |
| Weight bytes | $M_w=P\times s_w$, adjusted for metadata |
| Layers | $N_{\ell}$ |
| KV heads | $N_{\text{kv}}$ |
| Head dimension | $d_h$ |
| KV bytes per element | $s_{\text{kv}}$ |
| KV bytes per cached token | $2N_{\ell}N_{\text{kv}}d_hs_{\text{kv}}$ |
| Total cached tokens | $\sum_i T_i$ |
| Total KV memory | KV bytes/token $\times\sum_iT_i$ |
| Runtime and safety allocation | Measured with the target engine |
| Approximate linear FLOPs/token | $2P$, or active-compute estimate for MoE |
| Candidate VRAM | Must exceed the complete allocation |
| Candidate HBM bandwidth | Used for decode estimate |
| Candidate tensor throughput | Use the intended data type |
| Candidate interconnect | Required when sharding across GPUs |

The final selection rule is:

> First choose GPUs on which the complete workload fits; then choose among those GPUs using bandwidth, compute, interconnect, measured latency, throughput, availability, and cost.

Useful references:

- [NVIDIA GPU performance and arithmetic-intensity guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA Nsight Compute roofline documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Hugging Face explanation of KV-cache shapes and behavior](https://huggingface.co/docs/transformers/main/cache_explanation)
- [vLLM cache configuration and memory controls](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)

---

**Continue to:** [08-quantization](./08-quantization.md) for the most common way to relax the VRAM and bandwidth constraints, or branch into the architecture-focused chapters starting at [09-architecture-evolution](./09-architecture-evolution.md).
