# LLM Inference and Architecture Notes

A chapter-based reference covering two complementary topics:

1. **LLM inference** — prefill, decode, KV cache, compute vs memory, GPU sizing, quantization
2. **Modern transformer architecture** — the path from GPT-2 to Kimi K3, attention variants, KDA, LoRA, distillation
3. **JEPA representation learning** — embedding prediction, stopgrad, and EMA target encoders
4. **Cross-model KV transfer** — skipping repeat prefill when a related small-model session is promoted to a larger model

The notes are split into 14 chapters in `chapters/`, with `index.json` as the master index. **Read the JSON index first** to find which chapter covers the topic you want; the chapter files contain the actual content.

> The chapter files are intentionally split up, so opening one is cheaper than scrolling a monolithic document. When you add new material, append to the appropriate chapter (or create a new one) and update `index.json` accordingly.

## Quick start

```bash
# 1. Browse the index
cat index.json | python3 -m json.tool

# 2. Read by chapter
ls chapters/

# 3. Find a topic
python3 -c "import json; idx=json.load(open('index.json')); print(idx['topics']['kv-cache'])"
```

## Chapter map

| # | Chapter | Summary | Read time |
|---|---|---|---:|
| 00 | [Glossary](./chapters/00-glossary.md) | Abbreviations, symbols, math signs, units, operations/bytes/intensity/time | 5 min |
| 01 | [Inference Overview](./chapters/01-inference-overview.md) | Two-phase model, prefill vs decode table, compute-bound vs memory-bound, mental model | 8 min |
| 02 | [Prefill and Decode Mechanics](./chapters/02-prefill-decode-mechanics.md) | Why prefill is compute-bound, why decode is memory-bound, KV cache trade-off, 7B example | 10 min |
| 03 | [Batching, Scheduling, and vLLM](./chapters/03-batching-scheduling.md) | Batching math, continuous batching, vLLM internals, latency vs throughput | 12 min |
| 04 | [Memory Planning](./chapters/04-memory-planning.md) | Weight memory, KV-cache formula, MHA/GQA/MQA, GB vs GiB, 8B GQA worked example | 15 min |
| 05 | [Compute and FLOPs](./chapters/05-compute-flops.md) | 2P rule, prefill/decode FLOPs, memory hierarchy, CUDA vs Tensor Cores | 15 min |
| 06 | [Bandwidth, Arithmetic Intensity, and Roofline](./chapters/06-bandwidth-roofline.md) | ops:byte, roofline analysis, H100 295 FLOP/byte example, FLOPs/byte → user count | 18 min |
| 07 | [Choosing a GPU Systematically](./chapters/07-gpu-sizing.md) | 8-step selection procedure, GPU property priority, sizing worksheet, multi-GPU | 10 min |
| 08 | [Quantization](./chapters/08-quantization.md) | Why quantization helps, formats, KV-cache caveat, roofline shift | 5 min |
| 09 | [From GPT-2 to Kimi K3](./chapters/09-architecture-evolution.md) | The original @waterloo_intern article on the architectural progression | 30 min |
| 10 | [Modern Attention: MHA to KDA](./chapters/10-modern-attention.md) | Companion notes: Q/K/V, MHA/MQA/GQA/MLA, FlashAttention, DeltaNet, KDA, hybrid | 20 min |
| 11 | [Training and Adaptation](./chapters/11-training-adaptation.md) | LoRA, QLoRA, distillation, gradient averaging | 12 min |
| 12 | [Cross-Model KV-Cache Transfer](./chapters/12-cross-model-kv-transfer.md) | NVIDIA research: map a related small model's KV cache into a larger receiver, skip re-prefill, limits and serving implications | 12 min |

| 13 | [JEPA and Stopgrad](./chapters/13-jepa.md) | I-JEPA architecture, embedding prediction, stopgrad, PyTorch detach, EMA, collapse | 8 min |

**Suggested reading order for inference:** `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`
**Suggested reading order for architecture:** `09 → 10 → 11 → 13`

## Topic index (reverse lookup)

A subset of frequently searched topics. The full mapping is in `index.json` under `topics`.

| Topic | Chapter(s) |
|---|---|
| `prefill` | 01, 02, 05 |
| `decode` | 01, 02, 05, 06 |
| `kv-cache` | 00, 02, 04, 10 |
| `compute-bound` | 01, 02, 06 |
| `memory-bound` | 01, 02, 06 |
| `batching` | 02, 03, 06 |
| `vllm` | 03 |
| `vram` | 04, 05, 07 |
| `flops` | 05, 06 |
| `arithmetic-intensity` | 01, 06 |
| `roofline` | 06 |
| `bandwidth` | 05, 06, 07 |
| `gpu-selection` | 07 |
| `quantization` | 04, 08 |
| `mha` / `gqa` / `mqa` | 00, 04, 10 |
| `mla` | 09, 10 |
| `flashattention` | 02, 10 |
| `linear-attention` | 09, 10 |
| `deltanet` | 09, 10 |
| `kda` | 09, 10 |
| `moe` | 00, 04, 07, 09 |
| `kimi-k3` / `kimi-linear` | 09 |
| `lora` / `qlora` / `distillation` | 11 |
| `cross-model-kv-transfer` / `prefill-reuse` | 12 |

| `jepa` / `i-jepa` / `stopgrad` / `detach` / `ema` | 13 |

## How to extend the notes

When new material is added:

1. **Append to the relevant chapter** in `chapters/`. Each chapter has a self-contained scope (see the `topics` array in `index.json`).
2. **If a topic is genuinely new** (no chapter fits), create a new file like `chapters/14-new-topic.md`, using the next free number.
3. **Update `index.json`**:
   - add the new chapter to the `chapters` array
   - add or extend entries in the `topics` reverse index
   - bump the `version` and `lastUpdated` fields
4. **Update this README** with the new row in the chapter table and any new topic entries.

Because `index.json` is the entry point, the JSON file — not the markdown — is the source of truth for what exists. A consumer script should always read `index.json` before opening any chapter file.

## Sources

- `llm-inference-prefill-decode-notes.md` — contributed to chapters 00 through 08
- `xtract-@waterloo_intern.md` — contributed to chapters 09, 10, 11
- [Heo et al. (2026), *Cross-Model KV Cache Transfer in LLM Families*](https://arxiv.org/abs/2608.03893) — contributed to chapter 12

- [Assran et al. (2023), *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243) — contributed to chapter 13

## File layout

```
learning_inference/
├── README.md                    # this file
├── index.json                   # master index (source of truth)
├── llm-inference-prefill-decode-notes.md   # original (preserved)
├── xtract-@waterloo_intern.md   # original (preserved)
├── combined-llm-inference-notes.md        # previous combined version (preserved)
└── chapters/
    ├── 00-glossary.md
    ├── 01-inference-overview.md
    ├── 02-prefill-decode-mechanics.md
    ├── 03-batching-scheduling.md
    ├── 04-memory-planning.md
    ├── 05-compute-flops.md
    ├── 06-bandwidth-roofline.md
    ├── 07-gpu-sizing.md
    ├── 08-quantization.md
    ├── 09-architecture-evolution.md
    ├── 10-modern-attention.md
    ├── 11-training-adaptation.md
    ├── 12-cross-model-kv-transfer.md
    └── 13-jepa.md
```
# Learning-Inference
