# Chapter 03 — Batching, Scheduling, and vLLM

> How batching changes the decode math, what continuous batching means, and how vLLM orchestrates concurrent requests.

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

## What "make the bottleneck less limiting to system-wide performance" means

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

## Does batching make an individual request slower?

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

## Does vLLM perform this batching automatically?

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

## PagedAttention and KV-block allocation

### The problem: a request's KV cache grows one token at a time

Every transformer layer stores a key and value for every processed token. The KV bytes for one token in one layer are:

$$
M_{\text{KV/token/layer}}
=
2N_{\text{KV-heads}}d_hs_{\text{KV}}.
$$

The factor 2 represents one key and one value. A request with $T$ cached tokens across $N_{\text{layers}}$ layers needs approximately:

$$
M_{\text{request}}
=
2N_{\text{layers}}N_{\text{KV-heads}}d_hs_{\text{KV}}T.
$$

The difficulty is not only that this can be large. The server normally does not know the final value of $T$ when the request begins. One response may stop after 20 tokens while another continues for 2,000.

A naive allocator has two unattractive choices:

1. Reserve a contiguous region large enough for the maximum possible sequence length. Most requests finish early, so much of the reserved region is never used.
2. Reserve only the current size and repeatedly find a larger contiguous region as the sequence grows. This can require copying data and can fail even when enough total free memory exists, because the free space is split into separated holes.

Two kinds of waste matter:

- **Internal fragmentation:** unused space inside an allocation, such as reserving room for 2,048 tokens for a request that stops at 200.
- **External fragmentation:** enough total free memory exists, but it is divided into holes rather than one sufficiently large contiguous region.

KV-cache waste reduces the number of concurrent sequences that fit in GPU memory, which then reduces the batch size and serving throughput.

### The paging idea

PagedAttention divides the KV-cache pool into equal-sized **physical blocks**. Each block has room for the K/V data of a fixed number of token positions. A request's logical token sequence is also divided into **logical blocks** of the same token capacity.

The physical blocks belonging to one request do not need to be adjacent:

```text
Request A logical KV order

logical block:       0          1          2
tokens:             0–3        4–7        8–9
                      │          │          │
block table:          7          1          9
                      │          │          │
GPU block pool:     block 7    block 1    block 9
```

The request sees one logically continuous sequence, even though its KV data occupies scattered physical blocks. This resembles an operating system's virtual pages and physical page frames, but these are application-managed KV blocks; the phrase does not necessarily mean that KV data is being paged between GPU and CPU memory.

The small per-request mapping is called a **block table**:

$$
\text{block\_table}_r[b]=p,
$$

where:

- $r$ is the request;
- $b$ is the logical block number in that request;
- $p$ is the physical block ID in the shared KV pool.

### Address translation for one token

Let each block hold $B$ token positions. For logical token position $t$:

$$
b=\left\lfloor\frac{t}{B}\right\rfloor
$$

selects the logical block, and:

$$
o=t\bmod B
$$

selects the token's offset inside that block. The physical block is:

$$
p=\text{block\_table}_r[b].
$$

Conceptually, the cache address is then:

$$
\text{address}
=
\text{KV-pool base}
+p\times\text{block bytes}
+o\times\text{token-slot bytes}.
$$

Real layouts also include layer, KV-head, key/value, element, and device dimensions, but the logical-block-to-physical-block translation is the important idea.

### Numerical allocation example

Assume:

```text
block capacity = 4 token positions
Request A currently has 10 cached tokens
```

The required number of blocks is:

$$
N_{\text{blocks}}
=
\left\lceil\frac{10}{4}\right\rceil
=3.
$$

Suppose the allocator assigns physical blocks 7, 1, and 9:

$$
\text{block\_table}_A=egin{bmatrix}7&1&9\end{bmatrix}.
$$

The mapping is:

| Logical tokens | Logical block | Physical block |
|---|---:|---:|
| 0–3 | 0 | 7 |
| 4–7 | 1 | 1 |
| 8–9 | 2 | 9 |

Only the final block is partly filled: two of its four token slots are unused. If the request generates tokens 10 and 11, they fill the remaining slots in physical block 9. Token 12 requires one new free physical block, but the previous blocks are not moved.

For this toy allocation, the maximum unused capacity caused by block rounding is fewer than $B$ token slots per sequence. That is much smaller than reserving every request's full maximum context length, although smaller blocks require larger block tables and more allocation metadata.

### What happens during prefill

For a prompt of $T$ tokens:

```text
1. Scheduler calculates ceil(T / B) required logical blocks.
2. KV manager obtains that many free physical blocks.
3. It records their IDs in the request's block table.
4. Each layer writes the prompt tokens' K and V into the corresponding slots.
5. Attention kernels receive the block table so they can read the prompt in logical order.
```

The physical order can be 7, 1, 9 while the attention operation still observes token order 0, 1, 2, ..., 9.

### What happens during decode

For each generated token:

```text
1. Find the request's current last logical block.
2. If it has a free slot, write the new token's K and V there.
3. If it is full, allocate one free physical block and append its ID to the block table.
4. Give the new query and block table to the paged-attention kernel.
5. The kernel visits the mapped blocks and attends to the cached K and V in logical token order.
```

When the request finishes, its physical blocks can return to the free-block pool. Reference counts are needed when blocks are shared, because a shared block cannot be reclaimed until no request uses it.

### Does PagedAttention make the attention calculation smaller?

No. For a request with $T$ cached tokens, the new decode query still normally attends to all $T$ keys and reads the corresponding values:

$$
\operatorname{softmax}\left(\frac{q_tK_{0:t}^{\mathsf T}}{\sqrt{d_h}}\right)V_{0:t}.
$$

PagedAttention changes **where** those K/V vectors are stored and how they are located. It does not inherently:

- reduce the number of K/V elements per used token;
- shorten the context;
- change full attention into linear or sparse attention;
- eliminate KV-cache bandwidth;
- make one isolated request's arithmetic substantially smaller.

Its main benefit is reducing allocation waste and enabling flexible sharing. Better memory utilization lets the server retain more active sequences, form larger continuous batches, and obtain higher system throughput.

### Block size trade-off

A smaller token block size usually means:

- less unused space in the final block;
- more block-table entries;
- more allocator and kernel indexing overhead.

A larger block size usually means:

- fewer mappings and larger contiguous reads;
- more unused slots in partially filled final blocks;
- coarser-grained prefix sharing and allocation.

The best size is therefore an implementation and hardware trade-off, not a universal mathematical constant.

### Why Python's automatic growth does not remove this problem

Python can make a container appear to grow automatically:

```python
items = []
items.append(x)
```

However, Python has not removed memory management. The list implementation owns an underlying allocation. When more capacity is required, its allocator obtains more memory and may create a larger backing array and copy references into it. The language runtime hides those operations from the programmer.

The KV cache is different from an ordinary Python list in several important ways:

- its tensors live mainly in GPU memory rather than Python's CPU object heap;
- one token's cache covers many layers, KV heads, and vector elements;
- GPU kernels need known data types, strides, shapes, and addresses;
- repeatedly creating a larger tensor and copying the old KV cache would consume substantial bandwidth and time;
- many concurrent requests grow and finish independently.

A PyTorch/CUDA tensor allocation is normally fixed-size. Code can create what appears to be a larger tensor, for example by concatenation, but that commonly means allocating new storage and copying old and new data rather than extending the same storage freely in place.

Therefore the issue is not whether Python is dynamically typed and C++ is statically typed. The issue exists at the memory-allocation and GPU-kernel level:

```text
Python API
   ↓ hides orchestration details
fixed-size GPU storage and compiled kernels
   ↓ still require explicit allocation and addressing
Paged KV allocator and block table
```

vLLM can reserve one global GPU KV-cache pool and automatically manage it from Python-facing APIs. The user does not manually assign physical blocks, but the serving engine must still do so internally.

### Hashing and free-block allocation are different operations

Paged KV allocation does not require hashing merely to find free space. A KV manager can maintain a free-block queue or free list:

```text
need another block
    ↓
remove one physical block ID from the free queue
    ↓
append that ID to the request's block table
```

Hashing becomes important for **prefix caching**. It answers a different question:

> Has an identical full token-prefix block already been computed, so this request can reuse its K/V block?

Conceptually:

$$
\text{prefix hash}
=
H(\text{parent-prefix hash},\ \text{token IDs in block},\ \text{extra identity data}).
$$

The hash is used as a lookup key for matching cached prefixes. It does not contain the K/V values and does not itself choose an empty physical address.

```text
free list / allocator → finds an unused physical block
block table           → maps one request's logical blocks to physical blocks
prefix hash table     → finds an already-computed matching prefix block
```

### Relationship to the other three optimizations

These techniques operate at different layers of the serving system:

| Technique | Main question it answers |
|---|---|
| PagedAttention | Where should each request's growing K/V data be stored? |
| Prefix caching | Can a later request reuse already-computed K/V blocks for an identical prompt prefix? |
| Chunked prefill | How should a long prompt's prefill work be divided across scheduler iterations? |
| Speculative decoding | Can a cheaper method propose several tokens and let the target model verify them together? |

**Prefix caching** builds naturally on block-managed KV memory. If two requests begin with identical token blocks, the later request can point to compatible cached blocks instead of recomputing that prefix. Implementations use ownership/reference counts and an eviction policy. Current vLLM prefix caching hashes full token blocks and their preceding prefixes; partially filled blocks are generally not reusable cache entries. It saves prefill computation when prefixes repeat, but it does not accelerate ordinary decode after the shared prefix.

**Chunked prefill** is primarily scheduling, not memory address translation. Instead of allowing one 20,000-token prompt to occupy a huge prefill iteration, the scheduler might process it in smaller token chunks and interleave those chunks with existing decode requests. This can protect inter-token latency and combine compute-heavy prefill work with memory-heavy decode work.

**Speculative decoding** is primarily a decode-compute optimization. A smaller draft model or another proposal method suggests multiple future tokens. The target model verifies them in a more parallel operation and accepts a valid prefix of the proposals. Block allocation is still useful for storing accepted K/V states and reclaiming space associated with rejected or abandoned proposals, but paging and speculation solve different problems.

Primary references:

- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM PagedAttention kernel design](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM automatic prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM chunked-prefill optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/)

## Must requests wait, and how are their weights distinguished?

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

## What programming languages is vLLM written in?

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

## Important qualification: not all memory traffic is shared

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

A practical summary is:

- **Small batch:** lower queueing latency, but poor GPU utilization and lower total throughput.
- **Moderate batch:** better GPU utilization and throughput, often with an acceptable latency cost.
- **Very large batch:** high throughput may continue, but memory use, per-step execution time, and individual-request latency can become worse.

Therefore, batching is not automatically "faster" for every user. It makes the serving **system** more efficient, while its effect on one request depends on scheduling, batch size, model size, context lengths, and current server load.

---

**Continue to:** [04-memory-planning](./04-memory-planning.md) for the actual memory budgets involved, or [06-bandwidth-roofline](./06-bandwidth-roofline.md) for the bandwidth-vs-FLOP/s trade-off.
