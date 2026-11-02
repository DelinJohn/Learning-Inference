# Chapter 04 — Memory Planning: Weights, KV Cache, and Runtime

> How to estimate VRAM capacity, model-weight memory, KV-cache memory, and the activations/workspaces that round out the budget.

## Compute vs VRAM vs bandwidth

Compute requirement and memory requirement are related, but they are not the same quantity:

- **Compute** is the arithmetic performed, normally measured in FLOPs.
- **VRAM capacity** is the amount of data that must remain allocated, measured in bytes.
- **Memory bandwidth** is the rate at which those bytes can be delivered to the compute units, measured in bytes per second.

An LLM can fit in VRAM yet still run slowly because its weights cannot be delivered to the compute units quickly enough. Conversely, a GPU can have enormous compute throughput but be unusable for a model that does not fit in its VRAM.

## Units: GB versus GiB

Hardware and software sometimes report memory using different units:

$$
1\ \text{GB} = 10^9\ \text{bytes},
\qquad
1\ \text{GiB} = 2^{30}\ \text{bytes}.
$$

Always convert to the same unit before deciding whether a model fits.

## Step 1: calculate model-weight memory

For a simple model whose parameters all use the same storage type:

$$
M_{\text{weights}} = P \times s_w,
$$

where:

- $P$ is the number of parameters;
- $s_w$ is the storage bytes per parameter.

| Weight format | Raw bytes per parameter |
|---|---:|
| FP32 | 4 |
| FP16 or BF16 | 2 |
| FP8 or INT8 | 1 |
| INT4 | 0.5 |

Examples:

$$
7\text{B FP16} \approx 7 \times 10^9 \times 2 = 14\ \text{GB},
$$

$$
70\text{B INT8} \approx 70 \times 10^9 \times 1 = 70\ \text{GB}.
$$

These are raw estimates. Quantized models also store scales, zero points, packing metadata, and sometimes unquantized layers. The loaded representation can therefore be larger than $P$ multiplied by the nominal bit width. Check the actual checkpoint format and measure the loaded engine.

For a Mixture-of-Experts model, most or all expert weights may need to be stored even though only a subset of experts is active for each token. Therefore:

$$
\text{weight VRAM uses total stored parameters},
$$

while

$$
\text{per-token compute uses active parameter paths}.
$$

## Step 2: calculate KV-cache memory

For a conventional decoder-only transformer, the KV-cache bytes per cached token per request are approximately

$$
M_{\text{KV/token}}
=
2N_{\ell}N_{\text{kv}}d_hs_{\text{kv}},
$$

where:

- the factor $2$ represents keys and values;
- $N_{\ell}$ is the number of transformer layers;
- $N_{\text{kv}}$ is the number of key/value heads;
- $d_h$ is the dimension of each head;
- $s_{\text{kv}}$ is the number of bytes per KV-cache element.

Here, uppercase $N$ is just notation for "number of" or "count." It is not an additional value that must be multiplied separately:

$$
N_{\ell}=\text{number of layers},
\qquad
N_{\text{kv}}=\text{number of KV heads per layer}.
$$

For example, if a model configuration contains `num_hidden_layers = 32` and `num_key_value_heads = 8`, then

$$
N_{\ell}=32,
\qquad
N_{\text{kv}}=8.
$$

The complete substitution for BF16 KV entries with head dimension 128 is

$$
M_{\text{KV/token}}
=2\times32\times8\times128\times2\ \text{bytes}.
$$

### How to find the number of KV heads

For a Hugging Face-format model, open the model repository's `config.json` and look for

```json
{
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8
}
```

For this example:

$$
N_{\ell}=32,
\qquad
N_{\text{query heads}}=32,
\qquad
N_{\text{kv}}=8.
$$

It is a GQA model: groups of query heads share a smaller number of key/value heads. The relationship identifies the common attention variants:

| Configuration | Attention type |
|---|---|
| $N_{\text{kv}}=N_{\text{query heads}}$ | Multi-head attention (MHA) |
| $1<N_{\text{kv}}<N_{\text{query heads}}$ | Grouped-query attention (GQA) |
| $N_{\text{kv}}=1$ | Multi-query attention (MQA) |

### Are KV heads different from attention heads?

The phrase **attention heads** is often used loosely. In model configuration, `num_attention_heads` normally means the number of **query heads**, while `num_key_value_heads` means the number of distinct key heads and value heads.

Their tensor shapes are approximately

$$
Q\in\mathbb{R}^{B\times H_q\times T_q\times d_h},
$$

$$
K,V\in\mathbb{R}^{B\times H_{kv}\times T_{kv}\times d_h},
$$

where $H_q$ is the query-head count and $H_{kv}$ is the KV-head count.

In standard multi-head attention (MHA):

$$
H_q=H_{kv}.
$$

Each query head has its own corresponding key and value head.

In grouped-query attention (GQA):

$$
H_q>H_{kv}>1.
$$

Several query heads share one key/value head. If $H_q=32$ and $H_{kv}=8$, the group size is

$$
g=\frac{H_q}{H_{kv}}=\frac{32}{8}=4.
$$

Thus, every group of four query heads attends using the same cached key and value head. The queries remain different, so the four heads can still calculate different attention scores and outputs.

In multi-query attention (MQA):

$$
H_{kv}=1.
$$

All query heads share one key head and one value head.

Only keys and values from previous tokens are retained in the KV cache; old queries are not needed after their token has been processed. That is why KV-cache sizing uses $H_{kv}$, not $H_q$.

Compared with MHA, GQA reduces KV-cache capacity and bandwidth approximately in proportion to

$$
\frac{H_{kv}}{H_q}.
$$

For 32 query heads and 8 KV heads, the raw KV cache is approximately one quarter of the size it would have with 32 KV heads, assuming the other cache dimensions and data type stay the same.

The configuration can also be inspected with Transformers:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("MODEL_ID")

n_layers = config.num_hidden_layers
n_query_heads = config.num_attention_heads
n_kv_heads = getattr(config, "num_key_value_heads", n_query_heads)
head_dim = getattr(
    config,
    "head_dim",
    config.hidden_size // n_query_heads,
)

print(n_layers, n_query_heads, n_kv_heads, head_dim)
```

The fallback from `num_key_value_heads` to `num_attention_heads` is appropriate for many conventional older MHA configurations. It is not a universal architecture rule. If the field is missing for an unfamiliar model, check that model family's configuration documentation or implementation rather than guessing.

Some architectures—such as multi-head latent attention, hybrid attention/state-space models, or layers with different attention types—do not fit the simple uniform KV-head formula. In those cases, use the architecture-specific cache dimensions reported by the model implementation or inference engine.

For cached sequence length $T$ and $C$ equally sized concurrent sequences:

$$
M_{\text{KV,total}}
=
C T \times 2N_{\ell}N_{\text{kv}}d_hs_{\text{kv}}.
$$

When sequence lengths differ, use the total number of cached tokens instead:

$$
M_{\text{KV,total}}
=
\left(\sum_{i=1}^{C}T_i\right)
2N_{\ell}N_{\text{kv}}d_hs_{\text{kv}}.
$$

Use the number of **KV heads**, not necessarily the number of query heads. Grouped-query attention (GQA) and multi-query attention (MQA) reduce KV memory by using fewer KV heads. Sliding-window, local, hybrid, and multi-head latent attention architectures require architecture-specific calculations.

### Worked KV-cache example

Consider an illustrative 8B-class GQA model with:

- $N_{\ell}=32$ layers;
- $N_{\text{kv}}=8$ KV heads;
- $d_h=128$;
- BF16 cache, so $s_{\text{kv}}=2$ bytes.

Per cached token:

$$
2 \times 32 \times 8 \times 128 \times 2
= 131{,}072\ \text{bytes}
= 128\ \text{KiB}.
$$

For one 4,096-token sequence:

$$
128\ \text{KiB} \times 4096 = 512\ \text{MiB}.
$$

For 16 such concurrent sequences:

$$
16 \times 512\ \text{MiB} = 8\ \text{GiB}.
$$

If its 8 billion weights are stored in BF16, the raw weight estimate is 16 GB, or about 14.9 GiB. Weights plus this worst-case KV allocation already require about 22.9 GiB before runtime overhead. A nominal 24 GiB GPU would therefore be unsafe for this workload even though the weights alone fit.

Quantizing weights does not automatically quantize the KV cache. An INT4 model may have small weight storage while still using a BF16 or FP16 cache unless KV-cache quantization is configured separately.

## Step 3: include activations, workspaces, and runtime allocations

Inference also needs memory for:

- temporary activations;
- attention and matrix-multiplication workspaces;
- CUDA graphs and compiled kernels;
- communication buffers for multi-GPU execution;
- sampler outputs and request metadata;
- allocator fragmentation and framework/runtime state.

There is no reliable universal percentage for these allocations. They depend on model architecture, maximum batched tokens, maximum sequences, attention backend, quantization implementation, CUDA-graph capture sizes, and parallelism strategy.

Use the planning equation

$$
M_{\text{VRAM required}}
=
M_{\text{weights}}
+ M_{\text{KV}}
+ M_{\text{activations/workspaces}}
+ M_{\text{runtime}}
+ M_{\text{headroom}}.
$$

Do not plan to consume 100% of physical VRAM. First calculate weights and KV capacity, then validate runtime overhead by loading and benchmarking the actual model and engine. vLLM can profile available memory and allocate the remaining budget to KV-cache blocks; its `gpu_memory_utilization` and explicit `kv_cache_memory_bytes` controls determine the per-instance budget.

Useful references:

- [Hugging Face explanation of KV-cache shapes and behavior](https://huggingface.co/docs/transformers/main/cache_explanation)
- [vLLM cache configuration and memory controls](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)

---

**Continue to:** [05-compute-flops](./05-compute-flops.md) for the FLOPs side of the same workload, then [07-gpu-sizing](./07-gpu-sizing.md) to put weights + KV + FLOPs into a single GPU-selection workflow.
