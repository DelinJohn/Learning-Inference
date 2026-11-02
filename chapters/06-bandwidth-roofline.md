# Chapter 06 — Bandwidth, Arithmetic Intensity, and the Roofline

> How memory bandwidth and FLOP/s interact, what the ops:byte ratio really means, and how to use a roofline chart to read bottlenecks.

## What do memory bandwidth and FLOP/s each do?

**Memory bandwidth moves the data needed for calculation**, while **FLOP/s describes how quickly the compute units can perform the calculation**.

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

### Simple decode example

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

## Step 5: estimate whether compute or bandwidth limits speed

### GPU ops:byte ratio versus algorithm arithmetic intensity

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

### Example using the book's numbers

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

### A small example that explains the ratio

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

### How can one byte lead to many operations?

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

## Why this connects to batching and quantization

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

## Does FLOPs per byte determine how many users can be served?

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

---

**Continue to:** [07-gpu-sizing](./07-gpu-sizing.md) for a systematic GPU selection procedure that combines VRAM, FLOPs, and bandwidth.
