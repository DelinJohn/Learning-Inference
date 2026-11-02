# Chapter 08 — Quantization

> Why reducing numerical precision speeds up inference, which formats matter, and the gotchas around KV-cache quantization.

## Why quantization often speeds up decode

If decoding is limited by moving weights and KV-cache data, reducing their representation size reduces the number of bytes transferred.

For example, changing weights from FP16 to INT8 approximately halves their storage size:

$$
2\ \text{bytes per parameter}
\longrightarrow
1\ \text{byte per parameter}.
$$

This can almost halve weight-transfer time in an ideal bandwidth-dominated case. Actual speedup is lower or higher depending on dequantization overhead, kernel quality, hardware support, batching, and other system costs. KV-cache quantization similarly reduces cache capacity and bandwidth requirements.

## Common weight formats

| Weight format | Raw bytes per parameter |
|---|---:|
| FP32 | 4 |
| FP16 or BF16 | 2 |
| FP8 or INT8 | 1 |
| INT4 | 0.5 |

For an end-to-end view, see the model-weight memory formula and quantization caveat in [04-memory-planning](./04-memory-planning.md).

## Two important caveats

1. **Quantized models also store metadata.** Scales, zero points, packing metadata, and sometimes unquantized layers all add to the loaded size. The actual representation in memory is therefore usually larger than $P$ multiplied by the nominal bit width. Check the actual checkpoint format and measure the loaded engine.

2. **Quantizing weights does not automatically quantize the KV cache.** An INT4 model may have small weight storage while still using a BF16 or FP16 cache unless KV-cache quantization is configured separately.

## How quantization changes the roofline

Quantization mainly reduces bytes transferred. If useful arithmetic stays similar while the byte count falls, arithmetic intensity rises:

$$
I=\frac{F}{D}
\quad\longrightarrow\quad
I'=\frac{F}{D_{\text{smaller}}}.
$$

However, the appropriate GPU ops:byte ratio may also change because peak hardware compute throughput depends on the chosen data type. Dequantization and unsupported or inefficient kernels can prevent the simple theoretical improvement from becoming a real speedup.

For the full context see [06-bandwidth-roofline](./06-bandwidth-roofline.md#why-this-connects-to-batching-and-quantization).

---

**Continue to:** the architecture-focused chapters starting at [09-architecture-evolution](./09-architecture-evolution.md).
