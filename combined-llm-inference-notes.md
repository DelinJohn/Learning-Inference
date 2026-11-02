# Combined LLM Inference and Architecture Notes

This document combines two related sets of notes:

1. **LLM Inference: Prefill, Decode, and Memory** — practical reference on inference performance, KV cache, memory bandwidth vs compute, and GPU sizing.
2. **From GPT-2 to Kimi K3** — architectural evolution from early transformers to modern hybrid attention/MoE models, plus notes on LoRA, QLoRA, and distillation.

Both sections are preserved in full below.

---

# Part 1: Large Language Model (LLM) Inference: Prefill, Key–Value Cache, and Memory-Bound Decoding

## Equation abbreviations and symbols

This focused reference explains the technical abbreviations and symbols used in the equations. It intentionally does not list ordinary storage units such as KB, MB, or GB.

### Technical terms used in the equations

| Abbreviation | Full meaning | Plain-language meaning |
|---|---|---|
| GPU | Graphics processing unit | A highly parallel processor used for model computation |
| VRAM | Video random-access memory | General term for memory attached to or used by a GPU |
| HBM | High Bandwidth Memory | A high-throughput GPU-memory technology commonly used by data-center accelerators |
| LLM | Large language model | A model that processes and generates token sequences |
| KV | Key–value | The attention keys and values saved for earlier tokens |
| KV cache | Key–value cache | Stored keys and values reused during generation |
| Q | Query | What the current token uses to search or score cached keys |
| K | Key | Representation compared with a query |
| V | Value | Information combined according to the attention scores |
| QKV | Query, key, and value | The three main attention projections |
| MHA | Multi-head attention | Every query head has its own key and value head |
| GQA | Grouped-query attention | Groups of query heads share fewer KV heads |
| MQA | Multi-query attention | All query heads share one key head and one value head |
| MoE | Mixture of Experts | Architecture storing many expert networks but activating only a subset per token |

**HBM versus VRAM:** VRAM is the general term for GPU memory; HBM is a particular high-bandwidth memory technology. The weights may be stored in HBM, and HBM bandwidth limits how quickly they reach the compute units.

### Operation and intensity terms

| Abbreviation | Full meaning | Plain-language meaning |
|---|---|---|
| op | Operation | One arithmetic or logical action |
| FLOP | Floating-point operation | One arithmetic operation on floating-point numbers |
| FLOP/s or FLOPS | Floating-point operations per second | Floating-point compute rate |
| ops:byte | Operations-to-byte ratio | Hardware compute rate divided by hardware memory bandwidth |
| arithmetic intensity | Operations divided by bytes transferred | Amount of work an algorithm performs for each byte it moves |
| bandwidth | Bytes transferred per second | Speed at which memory supplies data |

`FLOPS` is sometimes used ambiguously. In these notes:

- **FLOP** means a count of work;
- **FLOP/s** means a rate of work.

For example:

$$
14\ \text{billion FLOPs}
$$

is an amount of computation, while

$$
100\ \text{TFLOP/s}
$$

is a processor's computation rate.

### Symbols used in formulas

| Symbol | Meaning in these notes |
|---|---|
| $P$ | Number of model parameters |
| $P_{\text{active}}$ | Parameters whose computation is active for one token, especially for MoE models |
| $B$ | Batch size when used as a mathematical variable |
| $C$ | Number of concurrent sequences in KV-capacity formulas |
| $T$ | Token count or sequence/context length in the local formula |
| $T_i$ | Token count belonging to request $i$ |
| $N_{\ell}$ or $N_{\text{layers}}$ | Number of transformer layers |
| $H_q$ | Number of query heads |
| $H_{kv}$ or $N_{\text{kv}}$ | Number of key/value heads |
| $d$ | Hidden dimension or model width |
| $d_h$ or $d_{\text{head}}$ | Dimension of one attention head |
| $s_w$ | Storage bytes for one weight parameter |
| $s_{kv}$ | Storage bytes for one KV-cache element |
| $F$ | Number of arithmetic operations, usually FLOPs |
| $D$ | Number of bytes transferred |
| $BW$ | Memory bandwidth in bytes per second |
| $C_{\text{peak}}$ | GPU peak compute throughput in operations per second |
| $I$ | Arithmetic intensity in operations per byte |
| $R_{\text{GPU}}$ | GPU hardware ops:byte ratio |
| $G$ | Number of GPUs or parallel shards in a local formula |
| $X$ | Input activation matrix |
| $W$ | Model weight matrix |
| $Y$ | Output activation matrix, usually $Y=XW$ |
| $Q,K,V$ | Query, key, and value tensors |
| $S$ | Attention-score matrix |
| $O$ | Attention output |
| $m,n,k$ | Matrix dimensions in a generic matrix-multiplication formula |

A symbol's definition applies to its local formula. For example, $C$ denotes concurrency in KV-cache sizing, but $C_{m\times n}$ later denotes the output matrix in a generic matrix-multiplication example.

### Mathematical signs used in the notes

| Sign | Meaning |
|---|---|
| $=$ | Exactly equal under the stated definitions |
| $\approx$ | Approximately equal; useful engineering estimate |
| $\sim$ | Scales like or is an order-of-magnitude estimate |
| $\propto$ | Is proportional to |
| $<$, $>$ | Less than, greater than |
| $\leq$, $\geq$ | Less/greater than or equal to |
| $\lesssim$, $\gtrsim$ | Approximately less/greater than or bounded by |
| $\sum_i T_i$ | Add the token counts from every request $i$ |
| $W^\top$ | Transpose of matrix $W$: rows and columns are exchanged |

### How operations, bytes, intensity, and time connect

These quantities answer different questions:

| Quantity | Question answered |
|---|---|
| FLOPs | How much arithmetic work is required? |
| Bytes | How much data must be stored or moved? |
| FLOP/s | How quickly can the GPU perform arithmetic? |
| Bytes/s | How quickly can the GPU move data? |
| FLOPs/byte | How much arithmetic is performed per byte moved? |
| Seconds | How long should the operation take? |

The basic relationships are

$$
\text{arithmetic intensity}
=
\frac{\text{FLOPs}}{\text{bytes transferred}},
$$

$$
\text{compute time}
\approx
\frac{\text{FLOPs required}}{\text{achieved FLOP/s}},
$$

$$
\text{memory time}
\approx
\frac{\text{bytes transferred}}{\text{achieved bytes/s}}.
$$

The slower of compute time and memory time usually forms the main lower bound on execution time.

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

## Why prefill is usually compute-bound

Suppose the prompt contains $T$ tokens and the hidden dimension is $d$. A linear projection has the form

$$
Y_{T \times d} = X_{T \times d}W_{d \times d}.
$$

This is a large matrix–matrix multiplication. A tile of $W$ loaded from GPU memory can be reused when processing many rows (tokens) of $X$. Consequently, the GPU performs a large amount of arithmetic for each byte loaded.

The keys and values are constructed using projections such as

$$
K = XW_K,
\qquad
V = XW_V.
$$

These are large matrix multiplications over all prompt tokens. The results are then written to the KV cache. Writing the resulting tensors consumes memory bandwidth, but it is normally not the dominant cost of the complete prefill pass.

Prefill attention also handles many queries at once:

$$
S = QK^\top,
\qquad
O = \operatorname{softmax}(S)V.
$$

Keys and values can be reused across many queries. Tiled implementations such as FlashAttention exploit this reuse and avoid repeatedly writing large intermediate attention matrices to high-bandwidth memory. This raises effective arithmetic intensity.

Therefore, "KV-cache construction is compute-bound" should be interpreted as:

> The prefill phase that computes the prompt's transformer activations and produces the KV cache is usually dominated by highly reusable matrix computations.

It should not be interpreted as saying that allocating or writing cache memory is itself compute-heavy.

## Why token generation is usually memory-bound

During ordinary autoregressive decoding, a sequence generates one new token at a time. A linear layer now behaves approximately like a matrix–vector multiplication:

$$
y_{1 \times d} = x_{1 \times d}W_{d \times d}.
$$

The model still has to read its large weight matrices, but at batch size one each weight is used for only one token. There is much less reuse than in prefill.

For a dense model with $P$ parameters stored in FP16:

$$
\text{weight bytes read per token} \approx 2P,
$$

and a rough estimate of the computation is

$$
\text{FLOPs per token} \approx 2P.
$$

Thus, at batch size one,

$$
\text{arithmetic intensity}
\approx
\frac{2P\ \text{FLOPs}}{2P\ \text{bytes}}
= 1\ \text{FLOP/byte}.
$$

This is very low relative to the compute-to-bandwidth ratio of modern GPUs. As a result, the arithmetic units often wait for weights to arrive from GPU memory.

### Example: a 7-billion-parameter model

Ignoring embedding details, tied weights, cache effects, and other implementation differences, an FP16 7B model has approximately:

$$
7 \times 10^9 \times 2
= 14 \text{ GB of weights}.
$$

Generating one token requires roughly

$$
2 \times 7 \times 10^9
= 14 \text{ billion FLOPs}.
$$

Reading roughly 14 GB to perform roughly 14 billion FLOPs again gives about 1 FLOP per byte. This is an illustrative estimate, not an exact per-token accounting.

## How the KV cache affects decoding

For each transformer layer, the KV cache stores the key and value vectors of previous tokens. For a new token, attention uses one new query against the cached keys and values:

$$
s = qK_{\text{cache}}^\top,
\qquad
o = \operatorname{softmax}(s)V_{\text{cache}}.
$$

If the current context has length $L$, the decoder must read keys and values corresponding to those previous tokens. Each cached element has little reuse because there is only one new query per sequence in a basic decode step.

Very roughly, for one attention head with dimension $d_h$ and FP16 cache entries:

$$
\text{K and V bytes read} \approx 4Ld_h,
$$

while the two main attention products perform approximately

$$
\text{attention work} \approx 4Ld_h\ \text{FLOPs}.
$$

This is again around 1 FLOP per byte, before considering implementation details. It is therefore normally bandwidth-bound.

As the context grows, KV-cache traffic grows approximately linearly with context length per generated token. At sufficiently long contexts, KV-cache reads can become a major or dominant part of decode latency.

## Why use a KV cache if reading it is expensive?

Without a KV cache, every decode step would recompute the keys and values of all previous tokens. That would repeatedly run transformer computations over the entire prefix.

The KV cache changes the trade-off:

- **Without caching:** repeat a large amount of computation at every step.
- **With caching:** compute each token's keys and values once, store them, and read them on later steps.

The cache therefore trades repeated computation for memory capacity and bandwidth. Reading cached values is much cheaper than recomputing the complete prefix, even though those reads contribute to the memory-bandwidth bottleneck.

## Why batching helps decode

Suppose $B$ sequences are decoded together:

$$
Y_{B \times d} = X_{B \times d}W_{d \times d}.
$$

The same weight tile can now be reused across $B$ sequences. In an idealized calculation, weight-related arithmetic intensity grows roughly with batch size:

$$
\text{arithmetic intensity} \propto B.
$$

Batching therefore moves decode toward better compute utilization and can eventually make some operations compute-bound. It usually improves total throughput, although waiting to assemble batches or serving a large batch can affect individual-request latency.

### What "make the bottleneck less limiting to system-wide performance" means

A **bottleneck** is the resource that currently sets the maximum performance of the complete system. Improving a resource that is not limiting the system may have little effect on the final result.

For example, suppose decode is waiting mostly for model weights to be transferred from GPU memory. Adding more arithmetic units would not help much: those units would also wait for the weights. A useful optimization must reduce weight traffic, reuse the transferred weights better, or increase available memory bandwidth.

Batching does this through reuse. With one request, a simplified linear layer is

$$
y = xW.
$$

The GPU loads $W$ and uses it for one input vector. With $B$ requests batched together, it computes

$$
Y = XW,
\qquad X \in \mathbb{R}^{B \times d}.
$$

The GPU can load a tile of $W$ once and use it for all $B$ rows of $X$. Therefore:

- the amount of **model-weight data** fetched is approximately unchanged for that batched forward pass;
- the amount of computation performed with those weights increases approximately by $B$;
- more tokens are produced per weight load;
- arithmetic intensity and GPU compute utilization increase.

Using the earlier dense FP16 approximation, batch size one gives

$$
\text{arithmetic intensity}
\approx
\frac{2P}{2P}
= 1\ \text{FLOP/byte}.
$$

For an idealized batch of $B$ requests sharing one weight load,

$$
\text{arithmetic intensity}
\approx
\frac{2PB}{2P}
= B\ \text{FLOPs/byte}.
$$

For example, at batch size 16, the same weight data can support roughly 16 times as much arithmetic as at batch size one. This does not make memory traffic disappear. It makes model-weight bandwidth **less limiting** because more useful work is obtained from each byte transferred.

Another way to express the throughput effect is:

$$
\text{batch 1: one weight load} \longrightarrow 1\ \text{token},
$$

$$
\text{batch 16: one shared weight load} \longrightarrow 16\ \text{tokens}.
$$

This primarily improves **system throughput**—the total number of tokens generated per second across all requests. It does not necessarily improve the latency of one request. A request may wait for a batch to form, and a larger batch takes more total work to execute.

### Does batching make an individual request slower?

It **can**, but it does not always do so. Two effects compete:

1. **Better GPU efficiency:** a batched operation reuses weights and uses more of the GPU, so processing each token can be more efficient.
2. **Batching overhead:** an individual request may wait for other requests, and the GPU must perform the work for every sequence in the batch before completing that decode step.

The distinction is between latency and throughput:

$$
\text{latency} = \text{time experienced by one request},
$$

$$
\text{throughput} = \text{total tokens produced per second across all requests}.
$$

Increasing batch size usually improves throughput, but after some point it can increase time per output token and therefore make each request feel slower. It can also increase time to first token if the server deliberately waits to form a batch.

Modern inference servers commonly use **continuous batching**. Instead of waiting for a fixed group of requests to finish together, the scheduler adds and removes sequences between decode steps. This reduces waiting and avoids allowing a long request to hold up every shorter request. Nevertheless, the server still chooses a latency–throughput trade-off through limits such as maximum batch size and scheduling delay.

### Does vLLM perform this batching automatically?

Yes. When multiple requests are active, vLLM's scheduler automatically selects sequences and combines their work into each engine iteration. This is commonly called **continuous batching** or **iteration-level batching**. A request that finishes can leave the batch, and a waiting request can be admitted during a later scheduling iteration; requests do not need to arrive simultaneously or have the same output length.

The application normally does not have to manually assemble a fixed tensor batch. It can submit concurrent requests to a vLLM server, and vLLM performs the scheduling internally. However, vLLM cannot create concurrency: if only one request is active, there are no unrelated requests to batch with it.

Important scheduler controls include:

- `max_num_seqs`: the maximum number of sequences processed in one iteration;
- `max_num_batched_tokens`: the maximum token budget for one iteration;
- the scheduling policy, such as first-come, first-served or priority scheduling;
- chunked-prefill settings, which control whether a large prefill can be divided across iterations.

In vLLM V1, chunked prefill is enabled by default when supported. Decode requests are prioritized, and unused space in the iteration's token budget can be filled with pending prefill work. This can combine memory-bound decode work with compute-bound prefill work and improve GPU utilization.

Therefore:

> vLLM automates continuous batching, but the operator still tunes its limits to choose the desired latency, throughput, and memory trade-off.

Official references:

- [vLLM optimization and chunked-prefill documentation](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM scheduler configuration](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)

### Must requests wait, and how are their weights distinguished?

The application does **not** need to deliberately hold a request or manually combine its tensors with other requests. If the server is idle, vLLM can start the request. If other requests are already active, the scheduler can add the new request at an appropriate engine-iteration boundary. This can introduce a small scheduling or queueing delay, but it is not the same as always waiting for a fixed-size batch to fill.

There is normally no need to distinguish different weights. Requests served by one model instance use the **same model parameters**. Suppose requests A, B, and C have current hidden-state vectors $x_A$, $x_B$, and $x_C$. The engine stacks them as separate rows:

$$
X =
\begin{bmatrix}
x_A \\
x_B \\
x_C
\end{bmatrix}.
$$

It then performs one batched multiplication with the shared weight matrix:

$$
Y = XW
=
\begin{bmatrix}
x_AW \\
x_BW \\
x_CW
\end{bmatrix}.
$$

Matrix multiplication does not mix the rows together. Row A in the output belongs to request A, row B belongs to request B, and so on. The scheduler retains this row-to-request mapping.

The requests also have separate:

- token IDs and positions;
- sequence lengths and attention metadata;
- output and sampling state;
- KV-cache blocks.

Attention requires additional isolation because one request must never attend to another request's tokens. vLLM's KV-cache manager and attention metadata map each sequence to its own logical cache blocks. The attention kernel uses those mappings and sequence boundaries to read only the keys and values belonging to the correct request.

Conceptually:

$$
q_A \text{ attends only to } K_A,V_A,
$$

$$
q_B \text{ attends only to } K_B,V_B.
$$

The shared component is the model weight matrix $W$, which is precisely what batching reuses. The request-specific components—hidden states, positions, and KV caches—remain logically separate.

If two requests use genuinely different base models, their weight matrices are different and they are generally routed to separate model instances rather than combined into this ordinary batch. Adapter-based serving, such as LoRA, can use specialized batching because the large base weights are shared while smaller request-specific adapter operations are tracked separately.

### What programming languages is vLLM written in?

vLLM is written primarily in **Python**, but it is not a pure-Python execution engine.

- **Python** implements most user APIs, model integration, scheduling, configuration, distributed orchestration, and serving logic.
- **CUDA and C++** implement compiled, performance-critical operations that run close to the GPU hardware.
- **Triton kernels** are commonly authored through a Python-like kernel language and compiled into GPU code; GitHub generally counts their source files as Python.
- The repository also contains some **Rust**, plus build and support code such as Shell and CMake.

The Python layer does not execute every scalar multiplication itself. It organizes work and invokes PyTorch operations, Triton kernels, or custom compiled CUDA/C++ operations. Conceptually:

$$
\text{Python scheduler and model code}
\longrightarrow
\text{PyTorch/Triton/custom operators}
\longrightarrow
\text{compiled GPU kernels}.
$$

Thus, "vLLM is written in Python" is broadly correct from an API and codebase perspective, while "its fast numerical work executes in compiled GPU kernels" explains how it achieves high performance.

Official references:

- [vLLM GitHub repository and language breakdown](https://github.com/vllm-project/vllm)
- [vLLM CUDA installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

## Calculating inference compute, VRAM, and GPU requirements

Compute requirement and memory requirement are related, but they are not the same quantity:

- **Compute** is the arithmetic performed, normally measured in FLOPs.
- **VRAM capacity** is the amount of data that must remain allocated, measured in bytes.
- **Memory bandwidth** is the rate at which those bytes can be delivered to the compute units, measured in bytes per second.

An LLM can fit in VRAM yet still run slowly because its weights cannot be delivered to the compute units quickly enough. Conversely, a GPU can have enormous compute throughput but be unusable for a model that does not fit in its VRAM.

### Units: GB versus GiB

Hardware and software sometimes report memory using different units:

$$
1\ \text{GB} = 10^9\ \text{bytes},
\qquad
1\ \text{GiB} = 2^{30}\ \text{bytes}.
$$

Always convert to the same unit before deciding whether a model fits.

### Step 1: calculate model-weight memory

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

### Step 2: calculate KV-cache memory

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

#### How to find the number of KV heads

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

#### Are KV heads different from attention heads?

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

#### Worked KV-cache example

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

### Step 3: include activations, workspaces, and runtime allocations

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

### Step 4: calculate the arithmetic work

For matrix multiplication

$$
C_{m\times n}=A_{m\times k}B_{k\times n},
$$

the usual estimate is

$$
\text{FLOPs} \approx 2mkn,
$$

because each multiply-accumulate is counted as one multiplication plus one addition.

#### Why matrix multiplication requires approximately $2mkn$ FLOPs

The dimensions mean:

- $A$ has $m$ rows and $k$ columns;
- $B$ has $k$ rows and $n$ columns;
- the result $C$ has $m$ rows and $n$ columns.

One output element is a dot product:

$$
C_{ij}
=
\sum_{r=1}^{k} A_{ir}B_{rj}.
$$

Calculating this element requires approximately:

$$
k\ \text{multiplications}
+ k\ \text{additions}
\approx 2k\ \text{FLOPs}.
$$

There are $m\times n$ elements in the output matrix, so

$$
(m\times n)\times(2k)
=2mkn\ \text{FLOPs}.
$$

Mathematically, a dot product starting with its first product has $k$ multiplications and $k-1$ additions, giving $mn(2k-1)$ operations. Performance engineering convention normally uses $2mkn$ because hardware executes multiply-accumulate operations and the difference becomes insignificant for large $k$.

For example, multiplying a $4\times3$ matrix by a $3\times2$ matrix gives a $4\times2$ result:

$$
m=4,\qquad k=3,\qquad n=2,
$$

$$
\text{FLOPs}\approx2\times4\times3\times2=48.
$$

For a dense transformer, a useful first-order estimate for the parameterized linear layers is

$$
\text{FLOPs per processed token} \approx 2P.
$$

#### Why a dense LLM uses approximately $2P$ FLOPs per token

Most LLM parameters belong to weight matrices. Consider one linear layer:

$$
y=xW.
$$

A **parameter** is normally one learned scalar number—one element such as $w_{ij}$ inside a weight matrix $W$. The parameter is data, not an operation. When the model processes a token, that number may be used in operations such as

$$
a\leftarrow a+x_iw_{ij}.
$$

Here:

- $w_{ij}$ is one learned parameter;
- $x_i$ is one input activation, not a parameter;
- $x_iw_{ij}$ is one multiplication;
- adding that product into accumulator $a$ is one addition.

Therefore, one participating weight parameter causes approximately

$$
1\ \text{multiplication}+1\ \text{addition}
=2\ \text{FLOPs}
$$

for one processed token. The parameter does not "contain two FLOPs"; FLOPs occur only when the stored parameter is used in arithmetic.

For a small example, let one token's activation vector contain three elements and let the linear layer produce two output elements:

$$
x=
\begin{bmatrix}
x_1 & x_2 & x_3
\end{bmatrix},
\qquad
W=
\begin{bmatrix}
w_{11} & w_{12}\\
w_{21} & w_{22}\\
w_{31} & w_{32}
\end{bmatrix}.
$$

$W$ contains six weight parameters. The two outputs are

$$
y_1=x_1w_{11}+x_2w_{21}+x_3w_{31},
$$

$$
y_2=x_1w_{12}+x_2w_{22}+x_3w_{32}.
$$

All six weights participate once. Hardware-performance convention treats this as approximately six multiply-accumulates:

$$
6\ \text{parameters}
\times2\ \text{FLOPs per participating parameter}
\approx12\ \text{FLOPs}.
$$

If additions are counted only between already-created products, the literal expression has six multiplications and four additions. The standard $2P$ engineering estimate counts a fused multiply-add as two FLOPs and ignores this small boundary difference.

If $W$ contains $p_W$ weights, processing one token uses each weight approximately once in a multiply-accumulate:

$$
\text{layer FLOPs per token}\approx2p_W.
$$

Adding the work from all the model's linear weight matrices gives

$$
2p_{W_1}+2p_{W_2}+\cdots
=2\sum_jp_{W_j}
\approx2P.
$$

Therefore, a dense 7-billion-parameter model performs roughly

$$
2\times7\ \text{billion}
=14\ \text{billion FLOPs}
$$

in its parameterized linear layers for one processed token.

The $2P$ rule is a first-order estimate, not an exact total. It does not fully count attention's context-dependent products, normalization, activation functions, embeddings, sampling, or implementation overhead. Some parameters may also be tied or unused in a particular operation. For a Mixture-of-Experts model, per-token compute depends mainly on the experts activated for that token rather than every stored expert parameter.

In particular, not every stored model parameter is necessarily used once for every token:

- an embedding table normally performs a row lookup rather than multiplying by every stored embedding value;
- a bias parameter usually contributes one addition rather than one multiply-add;
- normalization parameters use different operations;
- a Mixture-of-Experts model activates only selected expert parameters;
- attention's $QK^\top$ and probability-times-$V$ calculations add FLOPs that are not represented simply by counting learned parameters.

That is why $2P$ is a useful approximation for the dominant dense linear layers, not an exact law for every parameter in the model.

#### Prefill compute

For a prompt containing $T$ tokens:

$$
F_{\text{prefill,linear}} \approx 2PT.
$$

The factor $T$ appears because prefill processes $T$ prompt tokens. Each model weight participates in approximately one multiply-accumulate for each token:

$$
\underbrace{2P}_{\text{linear FLOPs for one token}}
\times
\underbrace{T}_{\text{prompt tokens}}
=2PT.
$$

For a 7B dense model and a 1,000-token prompt, the rough linear-layer estimate is

$$
2\times7\times10^9\times1000
=14\times10^{12}\ \text{FLOPs}.
$$

This is 14 teraFLOPs of **work**, not 14 teraFLOPs per second. FLOPs measure an amount; FLOP/s measures the rate at which a GPU performs that amount.

Attention adds a context-dependent term. For $N_{\ell}$ full-attention layers with hidden width $d$, a rough upper-style estimate for the two attention matrix products is

$$
F_{\text{prefill,attention}}
\sim
4N_{\ell}T^2d.
$$

Causal masking reduces useful work relative to a completely dense square attention matrix, and optimized kernels change the operations and memory traffic actually executed. Treat this as an order-of-growth estimate, not an exact benchmark predictor.

#### Decode compute

For a batch of $B$ active sequences generating one token each:

$$
F_{\text{decode,linear/step}}
\approx
2PB.
$$

If their current context lengths are $T_1,\ldots,T_B$, standard full attention adds approximately

$$
F_{\text{decode,attention/step}}
\sim
4N_{\ell}d\sum_{i=1}^{B}T_i.
$$

The linear-layer estimate is constant per new token, while attention work and KV-cache traffic grow with the existing context. For MoE models, replace total parameter count with an architecture-aware active-compute estimate; total stored parameters still matter for VRAM.

### If the model is already in VRAM, how is it used for compute?

"The model is in VRAM" means its weights remain allocated in the GPU's high-bandwidth memory rather than being repeatedly loaded from disk or system RAM. It does **not** mean the tensor cores can use every weight directly from its resting location at zero cost.

There are two different kinds of loading:

1. **Model startup:** checkpoint weights move from storage, usually through CPU memory, into GPU VRAM. This normally happens once when the model is loaded.
2. **Every forward pass:** weight tiles move from GPU VRAM through the GPU memory hierarchy to the arithmetic units. This happens repeatedly while serving tokens.

The simplified data path is

$$
\text{HBM/VRAM}
\longrightarrow
\text{L2 and on-chip caches}
\longrightarrow
\text{registers/shared memory}
\longrightarrow
\text{arithmetic units}.
$$

During each layer's matrix multiplication, the relevant weight tiles must be delivered through this memory hierarchy to the arithmetic units. Large LLM weights do not all fit in on-chip cache, so much of the model is streamed from HBM during each decode iteration.

The same weights therefore appear in three related calculations:

| Calculation | What is being measured | Approximate dense-model expression |
|---|---|---|
| VRAM capacity | Space occupied while the model remains loaded | $P\times s_w$ bytes |
| HBM traffic | Weight data delivered during a forward pass | Approximately $P\times s_w$ bytes per small-batch decode step |
| Arithmetic work | Multiplications and additions performed using the weights | Approximately $2P$ FLOPs per processed token |

These are not three separate arithmetic charges. They are three views of the same weights: stored data, moved data, and arithmetic performed with that data.

For one weight value $w$ and one activation value $x$, the arithmetic operation is conceptually

$$
a\leftarrow a+xw.
$$

Reading $w$ supplies the data. Multiplying $xw$ and adding it to $a$ supplies approximately two FLOPs. Merely keeping $w$ allocated in VRAM performs no arithmetic.

The FLOPs are already counted by matrix operations such as

$$
Y=XW.
$$

#### Dense 7B FP16 example

For an illustrative dense 7B model stored in FP16:

**Capacity:**

$$
7\times10^9\ \text{parameters}
\times2\ \text{bytes/parameter}
\approx14\ \text{GB of weights}.
$$

**Weight traffic for a small-batch decode step:**

$$
\text{approximately 14 GB read from GPU memory},
$$

assuming the dense weights are streamed roughly once and ignoring cache effects, KV traffic, and other tensors.

**Linear arithmetic for one token:**

$$
2\times7\times10^9
\approx14\ \text{billion FLOPs}.
$$

The model is not transferred from CPU memory for every token. It stays in GPU memory, but approximately 14 GB of weight data still has to travel from GPU memory toward the compute units during each low-batch forward pass.

#### Layers are computed sequentially, so why count all parameters?

Yes, transformer layers depend on one another and are normally executed in order:

$$
x
\xrightarrow{W_1}
h_1
\xrightarrow{W_2}
h_2
\xrightarrow{W_3}
\cdots
\xrightarrow{W_L}
y.
$$

The GPU is not multiplying every model parameter at exactly the same instant. During layer $\ell$, it operates on that layer's weights $W_{\ell}$. Even within one large matrix multiplication, the kernel usually works on smaller weight tiles that fit in registers or shared memory.

However, the token must eventually pass through every dense layer. Here $L$ is the total number of layers and $\ell$ is the index of one particular layer. If layer $\ell$ contains $P_{\ell}$ participating weight parameters, its approximate work is

$$
F_{\ell}\approx2P_{\ell}.
$$

The complete token's work is the sum across layers:

$$
F_{\text{token}}
\approx
\sum_{\ell=1}^{L}2P_{\ell}
=
2\sum_{\ell=1}^{L}P_{\ell}
\approx2P.
$$

Sequential execution changes **when** each operation happens, but not the total number of operations that must eventually happen.

The same distinction applies to memory:

| Memory level or quantity | What must be present |
|---|---|
| GPU VRAM capacity | Normally the weights of all layers for the loaded model |
| Weight data actively used at one moment | One layer, or part of one layer |
| Registers/shared memory | A small tile from the active layer |
| Total weight traffic for one dense decode pass | Roughly the sum of the weights read across every layer |

Suppose an illustrative 7B model had 32 similarly sized layers. The average layer would contain roughly

$$
\frac{7\ \text{billion}}{32}
\approx219\ \text{million parameters}.
$$

In FP16, that is roughly

$$
219\ \text{million}\times2\ \text{bytes}
\approx438\ \text{MB}
$$

per average layer. Only part of that layer is on chip at a given instant, but all 32 layers together still occupy roughly 14 GB of GPU VRAM, and processing one token eventually reads and computes through all 32 layers. Real layers are not exactly equal, and embeddings, output heads, and other parameters are additional architectural details.

#### Where are weights placed while the GPU computes?

The illustrative average above was about **438 MB per layer**, not an exact architectural value. The GPU uses all of those layer weights over the duration of the layer, but it does not copy the complete 438 MB layer into one separate 438 MB compute buffer.

Instead, a matrix-multiplication kernel divides the weight and activation matrices into small **tiles**. Conceptually, each tile follows this path:

$$
\underbrace{W_{\text{tile}}\text{ in HBM/VRAM}}_{\text{persistent model storage}}
\longrightarrow
\underbrace{\text{L2 cache}}_{\text{automatic on-chip cache}}
\longrightarrow
\underbrace{\text{shared memory/registers}}_{\text{active tile storage}}
\longrightarrow
\underbrace{\text{tensor or arithmetic cores}}_{\text{multiply-accumulate}}.
$$

The input-activation tile follows a similar route. Partial output results are accumulated, commonly in registers, and the completed output tile is eventually written back to GPU memory.

The roles are:

| Component | Role |
|---|---|
| HBM/VRAM | Holds the complete model and other large tensors |
| L2 cache | Automatically caches recently accessed GPU-memory data |
| Shared memory | Fast on-chip scratch space shared by threads in a thread block |
| Registers | Very fast per-thread storage for operands and partial results |
| Tensor/arithmetic cores | Execute multiplication and addition; they are compute units, not bulk storage |

On-chip cache, shared memory, and registers are much smaller than a model layer. The kernel therefore repeats this process:

1. Load a small weight tile and activation tile.
2. Perform many multiply-accumulate operations on those tiles.
3. Accumulate or store the partial output.
4. Replace the tiles with the next tiles.
5. Continue until the entire layer has been processed.

So there are two correct but different statements:

- **Across the complete layer operation:** roughly all 438 MB of illustrative layer weights must be consumed.
- **At one instant on chip:** only small tiles of those weights are present near the compute units.

VRAM is not merely unrelated passive storage. It is the GPU's large global memory, and GPU kernels read it directly through the GPU memory hierarchy. The CPU is not involved in these per-tile transfers when the model is fully resident on the GPU.

This tiled streaming is also why memory bandwidth matters. If the next weight tile cannot arrive quickly enough, the tensor cores wait. If one loaded tile can be reused across many prompt tokens or batched requests, the GPU performs more arithmetic before replacing it, increasing arithmetic intensity.

#### Is there a "CUDA layer," and should CUDA cores affect GPU selection?

There is no CUDA layer in the memory path. **CUDA** is NVIDIA's programming platform and software model. The relevant hardware components are commonly called **CUDA cores** and **Tensor Cores**:

- **CUDA cores** execute general arithmetic and many elementwise or control-heavy operations.
- **Tensor Cores** specialize in matrix multiply-accumulate operations at supported numerical formats.
- **HBM/VRAM, caches, shared memory, and registers** supply and hold the data used by those compute cores.

Most expensive dense LLM operations are matrix multiplications, so optimized inference normally attempts to use Tensor Cores. Other work—normalization, activation functions, indexing, sampling, and unsupported operations—may use ordinary CUDA cores or other specialized units.

The word **FLOPS** on a specification sheet is incomplete unless it states all of the following:

1. numerical format, such as FP32, BF16, FP16, or FP8;
2. ordinary-core or Tensor Core throughput;
3. dense or sparsity-assisted throughput;
4. the exact GPU form factor and power configuration.

For example, FP32 CUDA-core FLOP/s and FP16 Tensor-Core FLOP/s are different capabilities. An FP16/BF16 LLM matrix multiplication should be compared with the relevant dense Tensor Core figure, not an unrelated FP32 figure. A sparsity-assisted number should be used only if the model and kernel actually satisfy the required sparsity pattern.

CUDA-core count by itself is also a poor comparison across GPU generations. Core design, clock frequency, instruction throughput, Tensor Core generation, data-type support, memory system, and software kernels can differ.

For LLM inference, evaluate candidate GPUs in this order:

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

#### What do memory bandwidth and FLOP/s each do?

Yes: **memory bandwidth moves the data needed for calculation**, while **FLOP/s describes how quickly the compute units can perform the calculation**.

For a matrix multiplication $Y=XW$:

1. weight tiles from $W$ and activation tiles from $X$ are read from GPU memory;
2. they pass through caches into shared memory or registers;
3. Tensor Cores or other arithmetic units multiply and accumulate them;
4. output values from $Y$ are written to memory.

The two idealized times are

$$
t_{\text{memory}}
=
\frac{\text{bytes that must move}}
{\text{memory bandwidth in bytes/s}},
$$

$$
t_{\text{compute}}
=
\frac{\text{FLOPs that must be performed}}
{\text{compute throughput in FLOP/s}}.
$$

GPUs overlap data movement and arithmetic when possible, so a useful lower-bound model is

$$
t_{\text{operation}}
\gtrsim
\max(t_{\text{memory}},t_{\text{compute}}).
$$

The slower side limits the operation:

- if data arrives slower than the cores can consume it, the operation is memory-bound;
- if the cores take longer to process the supplied data, the operation is compute-bound.

##### Simple decode example

Suppose a dense FP16 model requires approximately 14 GB of weight traffic and 14 billion FLOPs for one token. Consider an idealized GPU with:

$$
BW=1\ \text{TB/s},
$$

$$
C_{\text{compute}}=100\ \text{TFLOP/s}.
$$

The ideal weight-transfer time is

$$
t_{\text{memory}}
=
\frac{14\ \text{GB}}{1000\ \text{GB/s}}
=0.014\ \text{s}
=14\ \text{ms}.
$$

The ideal arithmetic time is

$$
t_{\text{compute}}
=
\frac{14\times10^9\ \text{FLOPs}}
{100\times10^{12}\ \text{FLOP/s}}
=0.00014\ \text{s}
=0.14\ \text{ms}.
$$

In this simplified example:

$$
14\ \text{ms}>0.14\ \text{ms},
$$

so weight delivery is about 100 times slower than the arithmetic lower bound. The token is memory-bandwidth-bound, and buying substantially more FLOP/s without increasing effective bandwidth would provide little improvement.

Real inference also moves KV-cache data, activations, and outputs and includes kernel, scheduling, and communication overhead. The numerical example is intended to isolate the relationship between weight traffic and compute.

A compact analogy is:

- **VRAM capacity** is the warehouse size;
- **memory bandwidth** is the delivery rate from the warehouse;
- **FLOP/s** is the factory's calculation rate;
- **arithmetic intensity** is how much work the factory performs on each delivered byte.

Official references:

- [NVIDIA Tensor Core overview](https://www.nvidia.com/en-eu/data-center/tensorcore/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

It is possible to keep only some layers in VRAM and load other layers from CPU memory when needed. This is called **weight offloading**. It lowers required GPU capacity, but the weights must cross the much slower CPU-to-GPU interconnect during inference, usually making generation significantly slower.

Multi-GPU model parallelism can also lower weights per GPU:

- **Pipeline parallelism:** different GPUs store different groups of layers.
- **Tensor parallelism:** GPUs each store a shard of the weight matrices within layers.

The weight memory per GPU decreases, but the whole system still stores the model and must perform the complete computation. Communication between GPUs becomes an additional cost.

The approximate weight-only arithmetic intensity at batch size one is therefore

$$
\frac{14\ \text{billion FLOPs}}
{14\ \text{GB}}
\approx1\ \text{FLOP/byte}.
$$

At an idealized batch size of eight, the same weight tiles can be used for eight token rows:

$$
\frac{8\times14\ \text{billion FLOPs}}
{14\ \text{GB}}
\approx8\ \text{FLOPs/byte}.
$$

This is why batching makes weight loading less limiting: it performs more arithmetic with each weight byte fetched from HBM.

At small-batch decode, $W$ is large and has little reuse, so moving it from HBM is often slower than performing the arithmetic. With batching, a loaded weight tile is reused for multiple rows of $X$, which increases computation without proportionally increasing weight traffic.

#### Does FLOPs per byte determine how many users can be served?

It is an important factor, but it does not directly produce a user count. The sustainable number of active sequences is approximately constrained by both:

$$
C_{\text{usable}}
\leq
\min(C_{\text{memory}},C_{\text{performance}}).
$$

Here:

- $C_{\text{memory}}$ is how many sequences fit in the available KV-cache memory;
- $C_{\text{performance}}$ is how many sequences can be processed while still meeting throughput and latency targets.

Arithmetic intensity helps determine whether memory bandwidth or compute throughput sets $C_{\text{performance}}$. It is therefore a diagnostic input to capacity planning, not the final capacity answer.

For a simplified example, suppose:

- KV-cache capacity allows 100 active sequences;
- the measured server produces 1,000 output tokens per second;
- the application requires 20 output tokens per second for each active user.

The performance-based limit is

$$
C_{\text{performance}}
\approx
\frac{1000\ \text{tokens/s}}
{20\ \text{tokens/s/user}}
=50\ \text{active users}.
$$

The usable limit is therefore

$$
C_{\text{usable}}
\approx
\min(100,50)
=50.
$$

If KV memory allowed only 30 sequences, the memory-capacity limit would win instead. In production, prompt length, generated length, arrival pattern, prefill load, batching efficiency, and p95/p99 latency targets must also be included. "Users" should mean concurrently active generation sequences; idle registered users consume no decode capacity.

CPU weight offloading is different. If some weights are not resident in VRAM, they must cross the slower CPU-to-GPU interconnect during forward passes. This may enable a model to run with insufficient VRAM, but it normally increases latency substantially.

### Step 5: estimate whether compute or bandwidth limits speed

#### GPU ops:byte ratio versus algorithm arithmetic intensity

These quantities have the same unit, but describe two different things.

| Quantity | Describes | Formula | Meaning |
|---|---|---|---|
| GPU ops:byte ratio | Hardware | peak operations/s divided by peak bytes/s | How many operations the GPU can perform during the time required to deliver one byte |
| Arithmetic intensity | Algorithm or kernel | operations performed divided by bytes transferred | How much useful arithmetic the workload provides for every byte it moves |

The GPU's ratio is

$$
R_{\text{GPU}}
=
\frac{C_{\text{peak}}\ [\text{operations/s}]}
{BW_{\text{peak}}\ [\text{bytes/s}]}
\quad[\text{operations/byte}].
$$

It is a property of the chosen GPU **at a specific numerical format and operation type**. Changing from FP32 to FP16, BF16, FP8, or INT8 can change peak compute throughput and therefore change this ratio. The relevant memory bandwidth and whether sparse or dense compute figures are being used must also be stated.

The workload's arithmetic intensity is

$$
I_{\text{workload}}
=
\frac{F\ [\text{operations}]}
{D\ [\text{bytes transferred}]}
\quad[\text{operations/byte}].
$$

It is a property of a particular implementation of an operation. Batch size, tensor shapes, data type, kernel fusion, tiling, caching, and intermediate-memory traffic can all change it.

Although both ratios use operations per byte, their numerators and denominators differ:

$$
\underbrace{\frac{\text{operations/s}}{\text{bytes/s}}}_{\text{GPU capability}}
\quad\text{versus}\quad
\underbrace{\frac{\text{operations}}{\text{bytes}}}_{\text{workload behavior}}.
$$

The seconds cancel in the hardware ratio, allowing the two values to be compared directly.

The decision rule is

$$
I_{\text{workload}} < R_{\text{GPU}}
\quad\Longrightarrow\quad
\text{usually memory-bandwidth-bound},
$$

$$
I_{\text{workload}} > R_{\text{GPU}}
\quad\Longrightarrow\quad
\text{usually compute-bound}.
$$

Near equality, the workload is close to the roofline's ridge point, where compute and memory limits are approximately balanced.

#### Example using the book's numbers

Section 2.4.1 uses approximately 989 teraFLOPs/s of FP16 compute and 3.35 TB/s of memory bandwidth for its H100 example:

$$
R_{\text{GPU}}
=
\frac{989\times10^{12}\ \text{FLOPs/s}}
{3.35\times10^{12}\ \text{bytes/s}}
\approx
295\ \text{FLOPs/byte}.
$$

The result is **not 295 bits per second**. Notice how the seconds cancel:

$$
\frac{\text{FLOPs/s}}{\text{bytes/s}}
=
\frac{\text{FLOPs}}{\text{s}}
\times
\frac{\text{s}}{\text{bytes}}
=
\frac{\text{FLOPs}}{\text{byte}}.
$$

Therefore:

- $989\ \text{TFLOP/s}$ is a compute rate;
- $3.35\ \text{TB/s}$ is a memory-bandwidth rate;
- $295\ \text{FLOPs/byte}$ is a ratio between the two rates.

The value 295 means that, under the book's chosen specification assumptions, an algorithm needs to perform roughly 295 floating-point operations for each byte delivered from memory to offer enough work to use the GPU's full compute capability.

It does not describe the amount of memory transferred each second. The memory-transfer rate in the example is $3.35\ \text{TB/s}$.

This does **not** mean every H100 operation performs 295 FLOPs per byte. It means a workload needs arithmetic intensity near or above 295 FLOPs per byte, under the assumptions used for those peak specifications, to have enough work to approach that compute ceiling.

##### A small example that explains the ratio

Imagine a tiny processor with:

- compute capacity of $6$ operations per second;
- memory bandwidth of $2$ bytes per second.

Its hardware ops:byte ratio is

$$
R_{\text{hardware}}
=
\frac{6\ \text{operations/s}}
{2\ \text{bytes/s}}
=3\ \text{operations/byte}.
$$

Now consider three workloads.

**Workload A performs one operation per byte.** Memory can deliver two bytes each second, so the supplied data creates only

$$
2\ \text{bytes/s}\times1\ \text{operation/byte}
=2\ \text{operations/s}.
$$

The processor could perform six operations per second, but it receives only enough data-related work for two. It is memory-bound.

**Workload B performs three operations per byte.** Then

$$
2\ \text{bytes/s}\times3\ \text{operations/byte}
=6\ \text{operations/s}.
$$

Memory delivery and compute capacity are balanced. Both can be fully occupied in this idealized example.

**Workload C performs five operations per byte.** Memory could theoretically supply enough data for

$$
2\ \text{bytes/s}\times5\ \text{operations/byte}
=10\ \text{operations/s},
$$

but the processor can execute only six operations per second. Compute is now the bottleneck.

The hardware ratio is therefore a **balance point**, not a correctness requirement. A program with lower arithmetic intensity still runs correctly; it simply cannot use all the theoretical compute capacity.

Do not add useless operations to reach the balance point, and do not stop an algorithm from performing necessary operations above it. The algorithm's required useful work is fixed by what it must calculate; optimization changes how efficiently data is reused and how much unnecessary traffic or arithmetic is avoided:

- below the hardware ratio, reduce memory traffic or increase useful reuse;
- above the hardware ratio, reduce arithmetic work or use faster compute;
- near the ratio, compute and memory are approximately balanced.

The comparison uses operations per **byte**, not per bit, and the hardware ratio changes with GPU, numerical format, and operation type.

##### How can one byte lead to many operations?

Operations per byte does not mean that one byte stores that many parameters. It measures **reuse after data has been loaded**.

Also, the number of parameters stored per byte depends on the format:

- one FP16 or BF16 parameter occupies two bytes;
- one FP8 or INT8 parameter occupies one byte;
- two raw INT4 parameters fit into one byte, before metadata overhead.

The book's 295-ratio example uses FP16 compute assumptions. In FP16, one weight parameter $w$ occupies two bytes. Suppose it is loaded once and reused for three token activations:

$$
y_1=y_1+x_1w,
$$

$$
y_2=y_2+x_2w,
$$

$$
y_3=y_3+x_3w.
$$

There are three multiply-adds, or approximately six FLOPs. Looking only at this simplified weight traffic:

$$
I
=
\frac{6\ \text{FLOPs}}
{2\ \text{weight bytes}}
=3\ \text{FLOPs/byte}.
$$

The two-byte weight did not contain three calculations. It was loaded once, kept near the compute units, and used three times.

Extending the same simplified example, reuse across 295 tokens gives

$$
295\ \text{uses}
\times
2\ \text{FLOPs/use}
=590\ \text{FLOPs}.
$$

Dividing by the two bytes used to store the FP16 weight gives

$$
I
=
\frac{590\ \text{FLOPs}}
{2\ \text{bytes}}
=295\ \text{FLOPs/byte}.
$$

This is the reuse pattern behind high arithmetic intensity. In a real matrix multiplication, weight tiles are reused across many input rows and activation tiles are reused across many output columns. Real arithmetic-intensity calculations must count all relevant traffic, including activations and outputs, not only the simplified weight byte shown here.

For FP16 dense decode at batch size one, a weight is roughly used once for one token:

$$
\frac{2\ \text{FLOPs}}
{2\ \text{bytes}}
=1\ \text{FLOP/byte}.
$$

For prefill or batching, the same loaded weight is reused across many tokens, so the number of FLOPs grows faster than weight traffic and arithmetic intensity rises.

If a decode kernel has arithmetic intensity

$$
I_{\text{decode}}=62\ \text{FLOPs/byte},
$$

then

$$
62 < 295.
$$

The workload supplies only 62 FLOPs of work for each byte it transfers, while the hardware could perform arithmetic much faster relative to its data-delivery rate. The compute units therefore cannot be kept fully occupied; memory bandwidth limits performance.

Its ideal bandwidth roof would be

$$
3.35\ \text{TB/s}\times62\ \text{FLOPs/byte}
\approx
208\ \text{TFLOPs/s},
$$

which is well below the example's 989-TFLOPs/s compute ceiling.

If another kernel instead had arithmetic intensity of 400 FLOPs/byte, then

$$
400 > 295,
$$

so the GPU would reach its compute ceiling before exhausting the bandwidth-predicted performance. That kernel would tend to be compute-bound.

#### Why this connects to batching and quantization

Batching increases operations performed per model-weight load:

$$
I_{\text{decode weights}}
\approx
\frac{2PB}{2P}
=B\ \text{FLOPs/byte}
\quad\text{for idealized FP16 dense decode}.
$$

It therefore moves workload arithmetic intensity to the right on a roofline chart, making weight bandwidth less limiting.

Quantization mainly reduces bytes transferred. If useful arithmetic stays similar while the byte count falls, arithmetic intensity rises:

$$
I=\frac{F}{D}
\quad\longrightarrow\quad
I'=\frac{F}{D_{\text{smaller}}}.
$$

However, the appropriate GPU ops:byte ratio may also change because peak hardware compute throughput depends on the chosen data type. Dequantization and unsupported or inefficient kernels can prevent the simple theoretical improvement from becoming a real speedup.

One more subtlety is that "bytes transferred" must refer to a chosen level of the memory hierarchy. LLM roofline discussions commonly focus on HBM traffic, but an L2-cache or shared-memory roofline would use different bandwidth and traffic measurements. Use consistent boundaries when comparing the hardware and workload ratios.

For a candidate GPU, obtain its supported throughput at the intended data type and its HBM bandwidth. Define

$$
I_{\text{GPU}}
=
\frac{\text{peak FLOPs/s}}{\text{peak memory bytes/s}}.
$$

For the workload, estimate

$$
I_{\text{workload}}
=
\frac{\text{FLOPs}}{\text{bytes transferred}}.
$$

The roofline interpretation is:

- if $I_{\text{workload}} < I_{\text{GPU}}$, the workload tends to be memory-bandwidth-bound;
- if $I_{\text{workload}} > I_{\text{GPU}}$, it tends to be compute-bound.

Ideal lower bounds are

$$
t_{\text{compute}}
=
\frac{F}{C_{\text{effective}}},
\qquad
t_{\text{memory}}
=
\frac{D}{BW_{\text{effective}}},
$$

and

$$
t_{\text{operation}}
\gtrsim
\max(t_{\text{compute}},t_{\text{memory}}).
$$

Here $F$ is FLOPs, $D$ is bytes transferred, $C_{\text{effective}}$ is achieved compute throughput, and $BW_{\text{effective}}$ is achieved bandwidth. Real time is higher because of kernel launch, synchronization, communication, scheduling, and other overheads. Peak specification values are ceilings, not values that every model achieves.

For low-batch dense decode, a rough bandwidth model for one engine iteration is

$$
D_{\text{step}}
\approx
M_{\text{weights read}}
+ \sum_{i=1}^{B}M_{\text{KV read for request }i}.
$$

An approximate bandwidth-limited aggregate token rate is then

$$
\text{tokens/s}
\lesssim
\frac{B\,BW_{\text{effective}}}{D_{\text{step}}}.
$$

Equivalently, approximate bytes per generated token as

$$
D_{\text{per token}}
\approx
\frac{M_{\text{weights read}}}{B}
+ M_{\text{KV read per request}}.
$$

The first term falls as batching reuses weights. The second does not fall in the same way because each request has its own KV cache.

### Step 6: choose a GPU systematically

#### 1. Define the workload before selecting hardware

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

#### 2. Read the model configuration

Obtain at least:

- total and, for MoE, active parameter counts;
- number of layers;
- hidden dimension;
- number of query heads;
- number of KV heads;
- head dimension;
- maximum context or sliding-window behavior;
- supported numerical formats and inference kernels.

#### 3. Check VRAM capacity

Calculate:

$$
M_{\text{weights}}
+ M_{\text{KV at target concurrency/context}}
+ M_{\text{runtime and headroom}}.
$$

Reject any candidate that cannot fit this allocation reliably. If one GPU cannot fit it, consider quantization, reducing context or concurrency, KV-cache quantization, or multi-GPU sharding.

#### 4. Check prefill compute

Estimate prompt FLOPs and compare them with achieved—not merely peak—compute throughput. This determines whether the candidate is likely to meet the TTFT target.

#### 5. Check decode memory bandwidth

Estimate model-weight bytes per iteration and KV-cache bytes read at representative context lengths. Compare these with effective HBM bandwidth. This is often the most important check for low-batch token-generation speed.

#### 6. Check precision and software compatibility

Confirm that the GPU and serving stack efficiently support the desired:

- BF16, FP16, FP8, INT8, or INT4 path;
- attention backend;
- quantization method;
- model architecture;
- tensor/pipeline/expert-parallel implementation.

A format fitting in VRAM is not enough if the GPU lacks fast kernels for it.

#### 7. Check multi-GPU communication

Tensor parallelism can approximately divide many weight tensors across GPUs, but it adds collective communication inside transformer layers. Pipeline parallelism splits layers and has different latency and scheduling costs. Data parallelism creates model replicas and improves service capacity, but each replica must still fit its own model allocation.

For tensor parallel degree $G$, a rough starting point is

$$
M_{\text{weights per GPU}}
\approx
\frac{M_{\text{weights}}}{G},
$$

but KV cache, duplicated parameters, buffers, and workspaces are architecture- and engine-dependent. Do not assume total VRAM simply adds together with perfect efficiency. Interconnect bandwidth and topology—such as a fast GPU fabric versus PCIe alone—can determine whether adding GPUs improves or harms latency.

#### 8. Benchmark the real serving workload

Analytical calculations produce bounds and eliminate impossible candidates; they do not replace measurement. Benchmark with the real model, quantization, prompt-length distribution, output lengths, concurrency, and scheduler settings. Measure at least:

- peak VRAM use and cache capacity;
- TTFT;
- TPOT/ITL;
- aggregate input and output token throughput;
- p50, p95, and p99 latency;
- queueing, preemption, and out-of-memory behavior;
- performance at both normal and peak load.

### A compact GPU-sizing worksheet

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

A practical summary is:

- **Small batch:** lower queueing latency, but poor GPU utilization and lower total throughput.
- **Moderate batch:** better GPU utilization and throughput, often with an acceptable latency cost.
- **Very large batch:** high throughput may continue, but memory use, per-step execution time, and individual-request latency can become worse.

Therefore, batching is not automatically "faster" for every user. It makes the serving **system** more efficient, while its effect on one request depends on scheduling, batch size, model size, context lengths, and current server load.

### Important qualification: not all memory traffic is shared

The phrase "the same amount of memory traffic" is an intentional simplification. It mainly describes **model-weight traffic**.

Each sequence has its own activations and KV cache. When $B$ requests are batched, KV-cache reads generally also increase with $B$:

$$
\text{KV traffic} \propto B \times L,
$$

where $L$ represents context length. Unlike model weights, the KV cache of one request normally cannot be reused by unrelated requests.

Consequently, batching most directly reduces the importance of the **weight-bandwidth bottleneck**. At long context lengths, KV-cache bandwidth can remain limiting even with a large batch. At still larger batches, the workload may shift toward being compute-bound, or a different bottleneck such as inter-GPU communication may become dominant.

This illustrates a general inference-engineering pattern:

1. Identify the resource currently limiting end-to-end performance.
2. Increase its capacity, reduce demand for it, or reuse its data more effectively.
3. Measure the complete system again because another resource may become the new bottleneck.

## Why quantization often speeds up decode

If decoding is limited by moving weights and KV-cache data, reducing their representation size reduces the number of bytes transferred.

For example, changing weights from FP16 to INT8 approximately halves their storage size:

$$
2\ \text{bytes per parameter}
\longrightarrow
1\ \text{byte per parameter}.
$$

This can almost halve weight-transfer time in an ideal bandwidth-dominated case. Actual speedup is lower or higher depending on dequantization overhead, kernel quality, hardware support, batching, and other system costs. KV-cache quantization similarly reduces cache capacity and bandwidth requirements.

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

# Part 2: From GPT-2 to Kimi K3 — Architecture and Training

**Author:** ali (@waterloo_intern)
**Date:** July 27, 2026 at 08:52 PM

## 22580: From GPT2 to Kimi3, Explained

Twenty-two thousand five hundred and eighty. That's how many GPT-2 (2019) models fit inside KimiK3 (2026). We scaled up by a factor of 22,580 in seven years. But is it just... scale?

In this worklog, I'll walk through how we got here and how much, or how little, has actually changed since then. We'll trace the major architectural developments leading to KimiK3.

### GPT-2

GPT-2 is a decoder-only architecture:

The input receives token and positional embeddings:

Each transformer block, zoomed in, looks like this:

The attention process:

Once the final hidden-state matrix is produced, the language-model head maps it into vocabulary logits. During autoregressive decoding, only the logits at the final position are needed to select the next token.

> *This is an inefficiency of decoder-only generation: the model computes representations for every input position, but each decode step consumes only the final position's logits. Without caching, much of that work would be repeated for the next token.*

The KV cache comes from a straightforward observation: after appending the generated token to the input, the model would otherwise recompute projections for all previous tokens. Storing their key and value vectors avoids that redundant work.

That storage is the KV cache. It retains vectors for the previous N-1 tokens and can become large enough to create a memory-bandwidth bottleneck.

Overall, with about 50k possible tokens, 12 blocks, 12 heads, and an embedding dimension of 768, our baseline model is about 124M parameters.

At 2.8 trillion parameters, one KimiK3 model contains roughly as many parameters as 22,580 GPT-2 models.

### Linear Attention

Softmax attention applies its nonlinearity after the q·k product, coupling every query to every key. Linear attention instead applies a feature map, such as ELU+1, to q and k separately. This makes the product re-associable, so the growing set of K and V vectors can be folded into a fixed D×D state.

The paper's O(N²) framing threw me off. It's not true that "the cost per time-step for transformers scales with the square of the current sequence length". That's what Flash Attention fixes...  then I saw that it was released in 2020.

At the time, training commonly materialized the full N×N attention matrix, FlashAttention did not exist, and reference autoregressive implementations often recomputed the token history without a KV cache.

The same process is easier to see visually. Each decode step performs two ND reads and two 1D writes to HBM, while the KV cache grows linearly, in O(N), with the sequence length.

Notice the excessive reads and writes, which this paper replaces with:

There is a trade-off.

Here, we replace the exponential used by softmax with ELU+1 applied separately to q and k before they interact. Both approaches normalize the resulting scores, but the feature map used by linear attention is a less expressive approximation of the softmax kernel. That approximation can reduce fidelity, although the practical accuracy loss depends on the architecture and workload.

Notice that we still divide by the sum of qk, which is omitted from the diagram for simplicity. At a high level, attention consists of three steps:

1. Make the qk scores non-negative. Linear attention uses ELU+1, while softmax uses exponentiation.
2. Divide by the sum.
3. Compute the weighted average of the values.

This preserves the basic attention contract, but uses a less expressive feature map to make the QK scores non-negative.

### DeltaNet (Fast Weight Programmers)

A finite cache must overwrite or combine with information already stored. The state from token i-1 does not receive its own slot; it is added to the same D by D matrix. New queries can therefore no longer retrieve a perfectly isolated representation of each earlier token.

That addition is also the source of the efficiency gain. Updating the cache additively rather than by concatenation prevents it from growing in O(N), but the same operation causes information to interfere. DeltaNet addresses this loss of recoverability.

Eloquently put by Schlag's paper (Fast Weight Programmers): "when the sequence length exceeds storage capacity, the model may end up in an overcapacity regime. To properly operate under such a regime, the model should learn to dynamically interact with the memory contents and selectively decide which key-value associations to keep and which ones to delete. The purely additive instruction may be inappropriate for this purpose…. endlessly adding new associations to a memory of finite size, as in Eq. 17, inevitably will reach a limit."

The regime that makes linear attention attractive, where N is much larger than D, also exposes its main limitation. Once the state exceeds its effective capacity, associations begin to interfere because the update is additive and nothing leaves the cache.

A visual example makes this easier to follow.

Take a single association written as S = k.T @ v. If read back with the same key and you get k @ (k.T @ v), which is (k @ k.T) v, which is the squared norm of k times v. So read returns scaled by key's squared norm, and if normalize k to unit length, or just divide result by norm, get v back exactly.

Q is also a learned pointer. Wq and Wk read the same residual stream, and the query for a fact points at the key direction that fact was written into. The update first asks what information the current key retrieves from the cache. It subtracts that existing information from the value we want to store, multiplies the key by the difference, and adds the result back. Old information is removed and new information is written in its place.

### DeltaNet (Parallelizing Linear Transformers with Delta Rule)

This is the most difficult section of the post. It took me about seven hours to develop a working understanding of it, so I will build the explanation from the implementation. In short, DeltaNet implements a first-order linear recurrence with generalized Householder transition matrices, enabling chunk-wise parallel forward passes for hardware-efficient linear-time training. It splits the inputs and outputs into several chunks of size C, and computes outputs for each chunk based on the final state of the previous chunk and the query key value blocks of the current chunk.

The practical problem is prefill. A direct implementation of the Delta rule over a sequence of T tokens would look like this:

Unlike standard attention, this formulation requires a correction at every key vector, so the path to a parallel matrix multiplication is not immediately obvious. Even without the Delta rule, a direct linear-attention prefill remains sequential:

A chunked formulation provides a more efficient approach. The mechanics are easier to understand through an example:

Setting C=N recovers standard O(N^2) attention, while C=1 gives regular linear attention. Intermediate values we interpolate between trade additional within-chunk work for better hardware utilization. In practice, C is often 64 or 128 because tensor-core instructions operate efficiently at that granularity; UMMA is one example.

The intermediate tiles are folded into S as part of the state update:

Within a block, we do q(kᵀv). This is score first, the normal attention order with masking. Across blocks, we follow (kᵀv)q, so we're doing recurrent order, state first. Attention grows in O(N²) and this does not. Inside a block I do real attention (the masked QKᵀ times V), and across blocks I fold everything into the state and read it back with one matmul. So the cost splits in two. There's a fixed piece, 2Ld², which is the state work and doesn't care about C at all. And there's a growing piece, 2LCd, which is the score matrices sitting on the diagonal. Full attention is just the case where C equals L, and then that second term becomes 2L²d, quadratic. So the smaller I make C, the fewer FLOPs I do.

C=1 is the cheapest option in pure FLOP terms, but not necessarily in wall-clock time. A GPU can complete more arithmetic faster when the work maps efficiently onto its matrix-multiply hardware.

The next step is to extend the same approach to DeltaNet.

The underlying issue is simple: the chunking method used for purely additive attention does not directly apply to the delta updates:

We need every single state in order to compute the information that needs to be subtracted out. We can't parallelize it the same way without some mathematical re-parameterization. The authors therefore rewrite the delta updates from:

Here, a sequential loop computes one delta per iteration. The reparameterized form is:

This formulation allows the chunked code to compute all C deltas at once:

This gets us to our first comparison point: MHA vs DeltaNet Transformers:

### Gated Delta Net

We now have a method for making precise changes to the cache. With each new fact (each new key vector), we can look at exactly the old information stored at that point and replace it with the new information we want to attend to.

However, this mechanism can forget only an association for which it has a specific replacement. It cannot efficiently clear multiple associations during a context switch or decay memory generally to free capacity.

If we were doing purely additive linear attention:

Adding the ability to forget would be simple. We'd just need a parameter controlling the forgetful state:

This is the Mamba-2 contribution. We decay the previous cache, then add the new cache at full strength, preventing the state from growing without bound.

Uniformly decaying all key-value associations at each time step by a dynamic ratio is a working approach, and it's what Mamba does. But it doesn't account for the varying importance of different key-value associations.

That is, if the model needs to forget one specific association, all associations are forgotten equally. The Delta rule, in contrast, can update a single fact but has no way to make the rest of the facts decay.

So the Gated Delta rule combines Mamba's gated update rule with the Delta rule. It adds a parameter, alpha, that switches to the pure Delta rule when set to one and clears the memory when set to zero. The challenge is implementing this with the same parallel-chunks method.

The implementation uses the same DeltaNet reparameterization described in the previous section. The mathematics is nearly identical, with one addition: a data-dependent scalar between zero and one that controls the decay of the previous state. This combines effective key-value association learning with adaptive memory management.

The corresponding code changes are shown below:

The γʳ/γⁱ term accounts for cumulative decay. A token written at time step x and read at x+t has been multiplied by αₓαₓ₊₁αₓ₊₂…αₓ₊ₜ. This is the multiplicative analogue of a prefix-sum calculation.

The resulting architecture looks like this:

### KDA/Kimi Linear

At this point, researchers began experimenting with hybrid models that combine multiple forms of attention within one architecture, like Gated DeltaNet with Mamba.

Kimi Linear drew attention for one central claim: under controlled comparisons, it outperformed full attention. The authors presented it as a drop-in architectural replacement with better quality and up to 6x higher decode throughput.

Kimi Linear improves on Gated DeltaNet by introducing fine-grained gating. Instead of a single scalar decay, it learns a separate decay value for each channel.

The KDA update rule remains similar, but the code now looks more like this:

Here, alpha.reshape(nb, C, d) captures the paper's most significant contribution: fine-grained control over memory decay.

Placed beside the DeltaNet Transformer, the Kimi Linear architecture introduces three major changes:

1. It uses a hybrid system that interleaves Multi-head Latent Attention (MLA) layers.
2. It replaces the MLP with a Mixture-of-Experts (MoE) layer.
3. It adds capacity to DeltaNet through the alpha projection.

The later sections cover MLA and MoE in more detail. For now, the important point is that this is not blind scaling. The additional capacity has a specific mathematical purpose: the per-channel scale gives the model finer control over memory decay.

Scaling laws remain relevant, but capacity must be added in the right place and in a form the system can use. Each architecture in this progression adds capacity to address a concrete limitation in the preceding system.

### Kimi K3

Ultimately, the KimiK3 language backbone looks similar to the Kimi Linear model above. It contains 23 four-layer macrocycles. In each macrocycle, three layers use Kimi Delta Attention and the fourth uses Multi-head Latent Attention. The first layer uses a dense feed-forward network; every remaining layer uses a latent Mixture-of-Experts.

At first glance, the changes from Kimi Linear appear modest:

- A substantial increase in scale
- Blockwise AttnRes every 12 layers
- MLA query LoRA and output gating
- Latent-space MoE
- SiTU activations
- Gated MLA

KDA supplies constant-state recurrent memory, while periodic MLA layers retain full softmax retrieval over the context. The following simplified visualization provides a useful reference for the changes discussed below.

We will begin with the more direct changes: Gated MLA, latent-space MoE, and SiTU activations.

Gated MLA determines how much of each retrieved feature passes from MLA into the residual stream. It does this through element-wise multiplication with a gate projected from the input.

In a conventional MoE, a learned router uses dot-product similarity to send each token to a subset of expert networks. KimiK3 has 898 experts in total. Two are shared and process every token; of the remaining 896, the router selects 16 for each token.

KimiK3 also changes the expert activation. Instead of applying SiLU to the up projection, multiplying it element-wise by the gate, and then applying the down projection, it uses SiTU:

The model also down-projects inputs to the shared experts and up-projects their final sum:

This illustrates a recurring challenge in model inference. Without a fused kernel, the new activation is almost 3x slower than the original path. One offsetting optimization is that the experts operate in a compressed latent space, which makes their forward pass much faster and nearly halves the FLOPs.

The remaining changes are MLA query LoRA, output gating, and blockwise Attention Residuals every 12 layers. AttnRes adds roughly 2% inference latency, but provides two important benefits:

- Selective retrieval of earlier representations, which mitigates residual dilution and hidden-state growth
- A 1.25x compute advantage

AttnRes and MLA address the same underlying limitation from different directions. KDA layers operate with constant-size state and must inevitably discard information. MLA retrieves from the token context, while AttnRes retrieves from earlier depth-wise representations.

### AttnRes

Thanks to @chloey3k for help with this section. In each forward pass, the input passes through a stack of layers. Here, each layer consists of an attention block (KDA or MLA) and an MLP or MoE block. Normally, the input to each layer is the sum of the original embedding and every preceding layer's output, all weighted equally.

Here, h_i is the input to layer i, h_1 is the embedding of the current token (the last token in the sequence so far), and f_i(h_i) is the output of layer i (an attention or MLP block).

The problem is the lack of selective access. Different layer types receive the same aggregated state, even though they may benefit from different weightings. Because the recurrence is purely additive, later layers must also learn increasingly large outputs to influence the accumulated residual, which can destabilize training. Instead of treating all the layers equally, AttnRes multiplies each term of that sum by a specialized weight, which lets the model give more importance to whichever layers are most useful in context.

Each weight alpha_i is computed from a query-key dot product. The query is learned for each layer, while the keys and values come from earlier residual-stream states. The scores are normalized to sum to one, then used to form a weighted combination of those states.

The model therefore does not have to condition only on its immediate predecessor. AttnRes gives each layer selective access to earlier layer outputs, allowing its learned query to retrieve the representations most useful for the current computation.

The pseudocode below applies the same idea at block granularity. A block is the element-wise sum of the attention and MLP outputs accumulated across 12 decoder layers, stored as a single depth representation for later AttnRes mixing.

Applying residual attention at every layer would add too much training and inference cost. Applying it only at fixed block boundaries captures most of the benefit at a lower cost. In KimiK3, each boundary occurs after 12 decoder layers. Across 23 four-layer macrocycles, this produces eight AttnRes blocks, which increases our inference speed.

This is possibly the most important part of the block_attn_res function

This completes the progression from GPT-2 to KimiK3.

The central change is not scale alone. Each architectural step changes what the model stores, how it updates that state, or how it retrieves information that a fixed-size state cannot preserve.

KimiK3 combines constant-state recurrent memory, periodic softmax retrieval, sparse expert capacity, and selective depth-wise residual access. The result is a system that spends additional capacity where it has a specific functional role.

In essence, a fixed-capacity associative memory (fixed dimensions) needs an eviction policy, since a purely additive linear operation eventually adds interference once at capacity. To that end, learned selection, like gating, routing, or decay, is necessary, and attention is the most effective selective-read mechanism.

---

## Learning notes from our conversation

These notes preserve the important clarifications that are useful for understanding the article. They are intentionally selective rather than a transcript of every question.

### The central attention idea

For the current token:

- The **query** describes what information it is looking for.
- A **key** describes what information a previous token contains.
- A **value** is the information returned when its key matches the query.

Given the hidden-state matrix \(H\), the model learns three projections:

$$
Q = HW_Q, \qquad K = HW_K, \qquad V = HW_V
$$

Scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^{\mathsf T}}{\sqrt{d_{\text{head}}}}+M
\right)V
$$

Here:

- \(H\): token hidden states.
- \(W_Q,W_K,W_V\): learned query, key, and value projection matrices.
- \(d_{\text{head}}\): number of features in one attention head.
- \(M\): causal mask that prevents a token from reading future tokens.

The calculation has two conceptually separate stages:

```text
QKᵀ → determine how much attention each previous token receives
   ↓
multiply those attention weights by V → retrieve and mix information
```

During prompt processing, the queries, keys, and values for all prompt positions can be calculated in parallel under the causal mask. During cached generation, only the new token's query, key, and value are calculated. Previous keys and values are reused; previous queries are not needed again.

### Why attention is split into heads

The hidden state is projected and reshaped into several heads:

$$
[B,T,d_{\text{model}}]
\longrightarrow
[B,T,N_{\text{heads}},d_{\text{head}}]
$$

Each head creates its own attention distribution, so different heads can learn different relationships. The head outputs are concatenated and projected:

$$
\operatorname{MHA}(H)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_n)W_O
$$

The raw hidden vector is not merely chopped into unrelated pieces first. Learned projection matrices create the head-specific representations, and implementations normally calculate the projections using large fused matrix multiplications.

### Contextual Position Encoding

**Contextual Position Encoding (CoPE)** is still a type of position encoding. Ordinary absolute, relative, and rotary position methods measure distance using token positions. CoPE allows an attention head to learn which tokens should advance its position counter.

For query position \(i\) and earlier token \(j\), a simplified gate is:

$$
g_{i,j}=\sigma(q_i^{\mathsf T}k_j)
$$

The contextual distance is obtained by accumulating gates rather than blindly counting every token:

$$
p_{i,j}=\sum_{m=j}^{i}g_{i,m}
$$

Here:

- \(g_{i,j}\): how much token \(j\) should count as a positional step for query \(i\).
- \(p_{i,j}\): learned contextual position of token \(j\) relative to query \(i\).
- \(\sigma\): sigmoid function, producing a value between zero and one.

The contextual position changes the attention score:

$$
a_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}]
$$

The scores \(a_{i,j}\) are then passed through softmax and used to mix the values as usual. Fractional contextual positions are handled by interpolating nearby learned position embeddings.

For example, ordinary relative position counts every token:

```text
current token ← 1 token ← 2 tokens ← 3 tokens ← ...
```

A CoPE head can learn to count only sentence endings:

```text
current sentence  → contextual position 0
previous sentence → contextual position 1
sentence before   → contextual position 2
```

Another head could count words, code statements, paragraph boundaries, or some other learned feature. Therefore CoPE makes the meaning of position dependent on both the current query and the preceding content.

CoPE is part of the attention-score calculation, but it does not replace query/key/value attention, reduce the KV cache, or change full attention's sequence-length scaling. It is an architecture learned during training, not an inference option that can normally be enabled on an existing model. The original work evaluated relatively small models and explicitly did not establish results for billion-parameter language models, so CoPE should be viewed as a promising architectural method rather than a standard feature of deployed LLMs.

### The useful evolutionary map

```text
One fixed encoder vector
        ↓
Additive attention: decoder searches encoder states
        ↓
Dot-product attention: compare query and key by multiplication
        ↓
Transformer self-attention: tokens search tokens in the same sequence
        ↓
Multi-head attention: several learned searches at once
        ↓
Three efficiency branches
        ├── Reduce the KV cache: MQA → GQA → MLA
        ├── Restrict retrieval: local/sliding-window/sparse attention
        └── Compress history into state:
            Linear Attention → DeltaNet → Gated DeltaNet → KDA
```

This is not one perfectly straight historical chain. The branches solve different problems.

### Full softmax attention and its KV-efficient variants

In **Multi-Head Attention (MHA)**, each query head has its own key and value head:

$$
N_{\text{Q-heads}}=N_{\text{KV-heads}}
$$

In **Multi-Query Attention (MQA)**, every query head shares one key/value head:

$$
N_{\text{KV-heads}}=1
$$

In **Grouped-Query Attention (GQA)**, groups of query heads share key/value heads:

$$
1<N_{\text{KV-heads}}<N_{\text{Q-heads}}
$$

The progression is therefore:

```text
MHA → highest KV-cache size; separate K and V for every query head
GQA → fewer shared KV heads; quality/efficiency compromise
MQA → one shared KV head; smallest cache of these three
```

**Multi-Head Latent Attention (MLA)** compresses the key/value representation into a smaller latent vector:

$$
c_t^{KV}=h_tW_{\text{down}}
$$

Conceptually, keys and values can be obtained from this latent representation:

$$
k_t=c_t^{KV}W_{\text{up},K},
\qquad
v_t=c_t^{KV}W_{\text{up},V}
$$

MLA still performs global softmax retrieval over previous tokens. Its cache still grows with sequence length, but the stored representation per token is much smaller.

### KV-cache memory

The approximate key/value memory added by one token is:

$$
M_{\text{KV/token}}
=
2N_{\text{layers}}N_{\text{KV-heads}}d_{\text{head}}s_{\text{KV}}
$$

Here:

- The factor \(2\) represents one key and one value.
- \(N_{\text{layers}}\): number of transformer layers containing attention.
- \(N_{\text{KV-heads}}\): number of key/value heads in each layer. This is found in the model configuration; it is not necessarily the number of query heads.
- \(d_{\text{head}}\): features in each head.
- \(s_{\text{KV}}\): bytes used to store one key/value element.

For requests with sequence lengths \(S_i\), total KV-cache memory is approximately:

$$
M_{\text{KV,total}}
=
M_{\text{KV/token}}
\sum_i S_i
$$

The model weights are shared across all requests, but every request has its own KV cache.

### Exact prefill and decode sequence

For a prompt containing \(T\) tokens, each transformer layer calculates the prompt's queries, keys, and values largely in parallel:

$$
Q_{1:T},\qquad K_{1:T},\qquad V_{1:T}
$$

That layer stores only its prompt keys and values in the KV cache. It does not cache the previous queries or the attention-score matrix.

The final prompt query reads the prompt cache:

$$
o_T
=
\operatorname{softmax}\left(
\frac{q_TK_{1:T}^{\mathsf T}}{\sqrt{d_{\text{head}}}}
\right)V_{1:T}
$$

After the remaining layers and vocabulary projection, this produces the probability distribution used to choose the first generated token \(x_{T+1}\).

The generated token is then passed through the model. At each layer, the model calculates:

$$
q_{T+1},\qquad k_{T+1},\qquad v_{T+1}
$$

It appends the new key and value to that layer's cache:

$$
K_{1:T+1}=[K_{1:T};k_{T+1}]
$$

$$
V_{1:T+1}=[V_{1:T};v_{T+1}]
$$

The new query reads the complete cache:

$$
o_{T+1}
=
\operatorname{softmax}\left(
\frac{q_{T+1}K_{1:T+1}^{\mathsf T}}{\sqrt{d_{\text{head}}}}
\right)V_{1:T+1}
$$

This eventually predicts \(x_{T+2}\), and the process repeats:

```text
Prompt x₁ … xₜ
      ↓ prefill all prompt positions
Store K₁:ₜ and V₁:ₜ separately in every attention layer
      ↓
Last prompt position predicts xₜ₊₁
      ↓
Calculate qₜ₊₁, kₜ₊₁, and vₜ₊₁ at each layer
      ↓
Append kₜ₊₁ and vₜ₊₁ to that layer's cache
      ↓
qₜ₊₁ reads the complete KV cache
      ↓
Predict xₜ₊₂ and repeat
```

The important correction is that the first generated token is predicted by the last prompt position; that generated token is subsequently fed back into the model to predict the following token. The query performs the lookup, while the cached keys and values provide the memory being searched.

### FlashAttention is an implementation, not a new attention rule

FlashAttention calculates the same exact softmax-attention result using tiles. It avoids materializing the complete token-by-token attention matrix in high-bandwidth memory:

```text
Load small Q, K, and V tiles into fast on-chip memory
                        ↓
Calculate partial scores and the running softmax
                        ↓
Immediately combine with V and write the final output
```

Therefore:

```text
MHA/GQA/MLA/KDA → change the model architecture
FlashAttention  → changes how an attention calculation is executed
```

### Linear attention and fixed-size state

Softmax attention explicitly compares queries with previous keys. Linear attention uses a feature function \(\phi\) that allows the multiplication to be reassociated:

$$
S_t=S_{t-1}+\phi(k_t)v_t^{\mathsf T}
$$

$$
z_t=z_{t-1}+\phi(k_t)
$$

$$
o_t
=
\frac{\phi(q_t)^{\mathsf T}S_t}
{\phi(q_t)^{\mathsf T}z_t}
$$

Instead of storing a separate key and value for every previous token, it keeps the fixed-size state \(S_t\). This is efficient, but several associations can interfere because they share the same limited state.

### DeltaNet, Gated DeltaNet, and KDA

DeltaNet first asks what the current state already returns for a key:

$$
\widehat v_t=S_{t-1}^{\mathsf T}k_t
$$

It calculates the correction:

$$
e_t=v_t-\widehat v_t
$$

It then updates the association:

$$
S_t=S_{t-1}+\beta_tk_te_t^{\mathsf T}
$$

Here, \(\beta_t\) controls the strength of the write. Unlike purely additive linear attention, DeltaNet writes the difference between the desired value and what is already stored.

Gated DeltaNet adds a learned forgetting value \(\alpha_t\):

$$
\widetilde S_t=\alpha_tS_{t-1}
$$

It then applies the DeltaNet correction to \(\widetilde S_t\). This allows old memory to decay as well as allowing a specific association to be corrected.

**Kimi Delta Attention (KDA)** replaces the coarse scalar decay with channel-specific decay:

$$
S_t
=
\left(I-\beta_tk_tk_t^{\mathsf T}\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_tk_tv_t^{\mathsf T}
$$

$$
o_t=S_t^{\mathsf T}q_t
$$

Different state features can therefore forget at different rates. KDA does not preserve every previous key/value vector separately; it preserves their learned effect in a fixed-size state.

### Why hybrid attention is useful

```text
Full softmax/MLA
    + direct retrieval from any retained token
    - cache and attention work grow with context length

Linear attention/KDA
    + fixed-size recurrent state
    + efficient long-context decoding
    - history is compressed and exact facts can interfere
```

A hybrid model uses mostly KDA layers and periodic MLA layers. KDA carries efficient recurrent memory, while MLA provides occasional direct global retrieval. Kimi Linear uses a 3:1 KDA-to-MLA pattern.

### Choosing attention for a short-context SQL generator

A short context does not create a stronger reason to use KDA. It makes ordinary softmax attention cheaper, because the attention matrix and KV cache remain small. KDA's principal efficiency advantage appears at long context lengths, where it replaces a linearly growing KV cache with a fixed-size recurrent state.

For a specialized SQL generator, exact retrieval can be particularly valuable: the model may need to preserve precise table names, column names, joins, filters, and schema relationships. Compressing all of that information into a KDA state can introduce interference. For reliably short prompts, standard attention with GQA is generally the simpler starting point.

```text
Reliably short context + exact schema retrieval
    → standard softmax attention or GQA is usually sufficient

Very long context or many long concurrent requests
    → KDA or a KDA/MLA hybrid becomes more attractive
```

KDA is a model architecture, not an inference setting that can normally be enabled on an existing softmax-attention model. Replacing an existing model's attention layers with KDA changes the computation and usually requires architecture-aware training or substantial continued training. Reducing the configured context length alone does not convert the model to KDA.

### What an inference engineer needs to do with this knowledge

An inference engineer usually does not design KDA, MLA, or another new attention rule or train a model from scratch. The practical requirement is to recognize the architecture and predict how it will behave for the intended serving workload.

This knowledge is used to:

- Read the model configuration: attention type, query-head count, KV-head count, head dimension, layer count, precision, and supported context length.
- Estimate model-weight and KV-cache memory.
- Predict whether prompt processing, token generation, memory bandwidth, or capacity will be the likely bottleneck.
- Select an appropriate model for short or long contexts, exact retrieval requirements, concurrency, latency, and throughput.
- Confirm that the serving engine and GPU provide optimized kernels for that architecture.
- Benchmark the actual workload rather than relying only on theoretical complexity.

```text
Model-architecture researcher
    → invents or trains mechanisms such as KDA or MLA

Inference engineer
    → understands their consequences, selects a suitable model,
      configures the serving system, measures it, and optimizes deployment
```

Implementing attention mathematics from scratch is useful for learning and may be required for kernel or framework work, but it is not a routine requirement for deploying existing models.

### Complexity summary

Let \(T\) be the current context length and \(w\) a local-attention window size.

| Mechanism | Decode memory as \(T\) grows | Attention work for one new token | Main purpose |
|---|---:|---:|---|
| MHA | \(O(T)\), largest KV constant | \(O(T)\) | Full direct retrieval |
| GQA/MQA | \(O(T)\), smaller KV constant | \(O(T)\) | Reduce cache traffic |
| MLA | \(O(T)\), compressed per-token cache | \(O(T)\) | Preserve global retrieval with a smaller cache |
| Sliding-window attention | \(O(w)\) | \(O(w)\) | Bound work to recent context |
| Linear attention/KDA | Fixed-size recurrent state | Constant with respect to \(T\) | Efficient long-context memory |

The state operation in KDA still has a real cost determined by its state dimensions. "Constant" here means that it does not increase merely because the context contains more tokens.

### LoRA and QLoRA

**Low-Rank Adaptation (LoRA)** freezes an original weight matrix \(W\) and learns a low-rank correction:

$$
y=x(W+\Delta W)
$$

$$
\Delta W=BA
$$

Therefore:

$$
y=xW+xBA
$$

The original path \(xW\) and the adapter path \(xBA\) are both used in the forward pass. Backpropagation passes through the complete model so that the learning signal can reach earlier adapters, but only the LoRA matrices \(A\) and \(B\) are updated. The frozen base matrix \(W\) does not receive an optimizer update.

For:

$$
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}
$$

LoRA uses:

$$
A\in\mathbb R^{r\times d_{\text{in}}},
\qquad
B\in\mathbb R^{d_{\text{out}}\times r}
$$

The adapter contains:

$$
N_{\text{LoRA}}
=
r(d_{\text{in}}+d_{\text{out}})
$$

trainable parameters for that targeted matrix. The rank \(r\), selected target matrices, and selected layers determine the adapter size. Rank does not choose a small set of individual base weights; it controls the capacity of a correction spanning the targeted matrix.

**Quantized LoRA (QLoRA)** uses the same adapter idea but keeps the frozen base model quantized, commonly to 4-bit storage. The base weights are dequantized as needed for calculation, while the small LoRA matrices are trained in a higher-precision format.

Important limitation: LoRA and QLoRA save substantial training memory because gradients and optimizer states are not stored for every base parameter. They do not eliminate the base model's forward calculation or the need to propagate gradients through its computation.

### Distillation is separate from LoRA

Distillation decides **where the training target comes from**. LoRA decides **which student parameters are updated**.

```text
Prompt → large teacher model → teacher answer or probabilities
                                      ↓
Prompt → smaller student model → compare with teacher
                                      ↓
                                    loss
                                      ↓
                         update the student model
```

In response distillation, teacher-generated answers become training targets:

$$
\mathcal L_{\text{response}}
=
-\sum_t\log P_{\text{student}}
(y_t\mid y_{<t},x)
$$

In probability-distribution distillation, the student is trained to approach the teacher's complete next-token probability distribution:

$$
\mathcal L_{\text{distill}}
=
\operatorname{KL}
\left(
P_{\text{teacher}}\,\|\,P_{\text{student}}
\right)
$$

Here, \(\operatorname{KL}\) is Kullback-Leibler divergence, a measure of the difference between two probability distributions.

The teacher is normally used to create targets and is not updated. The student can be trained using either:

- Full fine-tuning, which updates the student's original weights.
- LoRA or QLoRA, which updates only adapters attached to the student.

Teacher answer generation is normally autoregressive and occurs token by token. Once a complete training sequence is available, the student calculates losses for all allowed sequence positions largely in parallel under the causal mask.

### Gradient averaging and large batches

The student does not normally update its parameters after every individual token. It combines the token losses in a batch, performs one backward pass, and then makes one optimizer update.

For per-token gradients \(g_i\), the batch gradient is:

$$
g_{\text{batch}}
=
\frac{1}{N}\sum_{i=1}^{N}g_i
$$

Averaging does not normally destroy accuracy. It reduces noisy, contradictory movement and finds a direction that lowers the mean loss over the batch. Summing and averaging contain the same directional information; their scale differs and is handled through the learning rate.

Large distributed models often use a large number of tokens to determine each update:

$$
B_{\text{global}}
=
B_{\text{per device}}
\times
N_{\text{devices}}
\times
N_{\text{gradient-accumulation steps}}
$$

The important distinction is:

```text
Many tokens contribute to one update
                  ≠
Every parameter receives a large numerical change
```

During full pretraining, billions of parameters may each receive a small update calculated from a very large global batch. During LoRA training, the batch calculation can be similar, but only the adapter parameters are updated.
