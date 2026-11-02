# Chapter 00 — Glossary: Abbreviations, Symbols, and Terms

> Reference tables used throughout the other chapters. Skim once, then return here when an equation or symbol is unclear.

## Technical terms used in the equations

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

## Operation and intensity terms

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

## Symbols used in formulas

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

## Mathematical signs used in the notes

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

## How operations, bytes, intensity, and time connect

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

---

**See also:** [04-memory-planning](./04-memory-planning.md) for the GB-vs-GiB unit note, and [06-bandwidth-roofline](./06-bandwidth-roofline.md) for the ops:byte ratio in context.
