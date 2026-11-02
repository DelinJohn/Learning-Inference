# Chapter 05 — Compute and FLOPs

> Where the FLOPs come from in a transformer, why the 2P approximation works for dense layers, and how weights move from VRAM to the tensor cores.

## Why matrix multiplication requires approximately $2mkn$ FLOPs

For matrix multiplication

$$
C_{m\times n}=A_{m\times k}B_{k\times n},
$$

the usual estimate is

$$
\text{FLOPs} \approx 2mkn,
$$

because each multiply-accumulate is counted as one multiplication plus one addition.

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

## Why a dense LLM uses approximately $2P$ FLOPs per token

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

## Prefill compute

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

## Decode compute

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

## If the model is already in VRAM, how is it used for compute?

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

### Dense 7B FP16 example

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

## Layers are computed sequentially, so why count all parameters?

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

## Where are weights placed while the GPU computes?

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

## Is there a "CUDA layer," and should CUDA cores affect GPU selection?

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

Official references:

- [NVIDIA Tensor Core overview](https://www.nvidia.com/en-eu/data-center/tensorcore/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

It is possible to keep only some layers in VRAM and load other layers from CPU memory when needed. This is called **weight offloading**. It lowers required GPU capacity, but the weights must cross the much slower CPU-to-GPU interconnect during inference, usually making generation significantly slower.

Multi-GPU model parallelism can also lower weights per GPU:

- **Pipeline parallelism:** different GPUs store different groups of layers.
- **Tensor parallelism:** GPUs each store a shard of the weight matrices within layers.

The weight memory per GPU decreases, but the whole system still stores the model and must perform the complete computation. Communication between GPUs becomes an additional cost.

---

**Continue to:** [06-bandwidth-roofline](./06-bandwidth-roofline.md) for the bandwidth side of the same workload, and [07-gpu-sizing](./07-gpu-sizing.md) to put FLOPs and bytes into a single selection process.
