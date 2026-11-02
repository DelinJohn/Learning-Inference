# Chapter 12 — Cross-Model KV-Cache Transfer

> A smaller model has already read a long conversation. Can a larger related model continue from that work without reading the whole conversation again? This chapter explains the answer proposed in [Heo et al. (2026)](https://arxiv.org/abs/2608.03893): transform the small model's KV cache into an approximation of the large model's KV cache.

## Start with the normal model switch

Imagine an assistant that uses a small, inexpensive model for ordinary requests and escalates difficult requests to a larger model.

The conversation has already accumulated 20,000 tokens. The small model has read all 20,000 tokens, so it owns a KV cache for them. The large model has not read them yet.

At this point, the large model cannot normally use the small model's cache. A KV cache is not merely a copy of the prompt. It contains the keys and values computed by the **particular model** at every layer.

```text
Same prompt text
     │
     ├── small model ──> small model's KV cache
     │
     └── large model ──> large model's KV cache
```

Even when the models are in the same family, they can have different numbers of layers and different internal representations. Therefore, the standard handoff is expensive:

```text
small model has processed the conversation
                 │
                 ▼
large model reads the full conversation again (re-prefill)
                 │
                 ▼
large model finally begins decoding
```

For a long history, this second prefill can dominate the handoff latency.

## The idea: translate the cache, not the text

The paper asks whether related models represent the same prompt in sufficiently similar ways that one cache can predict the other.

Its answer is: often, to a useful approximation. Instead of asking the large model to reprocess every prompt token, run a learned transformation over the small model's already-computed KV cache:

```text
prompt
  │
  ▼
small model prefill ──> source KV cache
                              │
                              │ mapper
                              ▼
                         target KV cache
                              │
                              ▼
                    large model starts decoding
```

The target cache is an **approximation** of the cache that the large model would have created itself. If that approximation is good enough, the larger model can continue the conversation without re-prefill.

## A short reminder: what is being transferred?

At each transformer layer, every past token contributes a key \(K\) and a value \(V\). When generating the next token, an attention head uses its new query \(Q\) to:

1. compare \(Q\) with the cached keys to decide which past tokens matter;
2. use the resulting weights to combine the cached values.

So a wrong cache can make the larger model attend to the wrong history, even if the prompt text was correct. That is why a cache handoff has to preserve useful attention behavior, not just match tensor shapes.

## Why a linear map might work

Models from the same family are trained similarly and often learn related features. The features are not identical, but one model's representation may be approximately expressible as a weighted combination of another's.

For one target layer and one target KV head, the basic question is:

> Given the source model's cached vectors for a token, can we predict the target model's cached vector for that same token?

The paper uses this familiar linear form:

$$
\text{predicted target vector}
=
\text{source features}\times W+b.
$$

There is one map for keys and another for values:

$$
\hat K_t^{l,h}=X_K^lW_K^{l,h}+b_K^{l,h},
\qquad
\hat V_t^{l,h}=X_V^lW_V^{l,h}+b_V^{l,h}.
$$

Here \(l\) is a target layer, \(h\) is a target KV head, \(X\) is information taken from selected source layers, and \(\hat K_t,\hat V_t\) are the cache vectors produced for the target model.

This is a linear transformation at handoff time. It does **not** update the weights of either LLM.

## How the mapper is learned

Before deployment, run the source and target models on the same calibration prompts. For every calibration token, collect:

```text
input:  source model KV features for that token
label:  target model KV features for the same token
```

Then solve for \(W\) and \(b\) with ridge regression:

$$
W^*=\left(X^\top X+\lambda I\right)^{-1}X^\top Y.
$$

Ridge regression is linear regression with a small penalty on very large weights. The penalty makes the solution more stable when source features are correlated.

At serving time, no gradient descent runs. The system simply applies the saved matrices to the source cache.

```text
Offline, once per source → target pair:
calibration prompts → collect both caches → fit mapper → save mapper

Online, at each promotion:
source cache → matrix multiplications → target-format cache → target decode
```

## Three details that make the simple idea work

### 1. A target layer can use several source layers

It is tempting to map source layer 10 only to target layer 10. That is often too simplistic: a 14B model and a 32B model may place comparable information at different depths.

For each target layer, the method selects the \(k\) source layers that best predict it, concatenates their cache features, and maps that combined input to the target layer. This cross-layer selection was the largest contributor in the paper's ablations.

### 2. Keys need special treatment because of RoPE

Many modern LLMs apply Rotary Position Embeddings (RoPE) to keys. RoPE mixes positional information into a key vector by rotating it according to its token position.

If a mapper learned directly from position-rotated keys seen near position 500, it may not transfer cleanly to position 20,000. The paper instead:

```text
source key with RoPE
        │
        ▼
remove source RoPE rotation
        │
        ▼
map position-free key content
        │
        ▼
apply the target model's RoPE rotation at the actual token position
```

Values do not carry this RoPE rotation, so they can be mapped directly.

### 3. Good vector matching is not enough

Two mapped caches can have a respectable numerical fit but still harm generation if their errors affect the few attention directions that matter most. The paper found that attention-output similarity predicted downstream quality better than raw KV \(R^2\) alone.

Practical lesson: evaluate the final task quality after handoff. Do not approve a mapper only because its cache tensors look numerically close.

## What “without changing anything” means—and does not mean

The appealing part of this method is that the source and target LLMs remain frozen. There is no LLM fine-tuning and no modification of the prompt.

But it is not a universal, zero-setup cache conversion:

| Unchanged | New work or artifact |
|---|---|
| Source-model weights | A source→target mapper is fitted offline |
| Target-model weights | Calibration prompts and both models' KV caches are required to fit it |
| Prompt and token history | The serving system must install target-format KV tensors correctly |
| Target model's normal decode | Mapper memory and matrix-multiply latency must be budgeted |

The paper used 500 calibration sequences of 1,024 tokens. Its reported mappers were roughly 4–12 GB for the evaluated pairs. Thus, the mapper is simple in *function* (linear), but it is not necessarily small in storage.

## What the paper found

The evaluation covered six pairs in the Qwen3, Llama 3.1, and Ministral 3 families. It was limited to pairs with matching KV-head counts and per-head dimensions.

- Four pairs retained 73–98% of the larger receiver's standalone-prefill accuracy, averaged across five benchmarks.
- Two pairs degraded sharply, showing that family membership alone does not guarantee a successful transfer.
- On Qwen3 14B ↔ 32B, mapping was 2.7–25× faster than asking the receiver to prefill again, depending on sequence length.
- A nonlinear MLP improved difficult pairs by up to 37 percentage points on HellaSwag, but that is no longer the paper's simple linear solution.

Treat these as pair-specific research results, not as a promise for arbitrary models or checkpoints.

## When this is useful

The clearest use case is a **cost–quality cascade**:

```text
1. Start on a small model for low cost.
2. Detect that the request needs a stronger model.
3. Transform the small model's existing cache.
4. Let the larger model continue from the transformed cache.
```

It is most attractive when the history is long and promotion happens often enough that avoiding repeated target-model prefill outweighs mapper storage and integration cost.

It does not eliminate:

- target-model weight memory;
- target-format KV-cache memory after the handoff;
- target-model decode cost;
- the need to validate quality for each source→target pair and workload domain.

## Deployment checklist

Before relying on cross-model KV transfer, answer these questions:

1. Are the source and target sufficiently related, and is their cache layout supported by the mapper?
2. Does the mapped-cache handoff preserve the quality metrics that matter for this application?
3. Is end-to-end promotion faster after including source-cache movement, mapping, and target-cache installation?
4. Does GPU memory have room for the mapper, the target model, and the target-format cache at expected concurrency?
5. What is the safe fallback? Usually: discard the mapped cache and run ordinary target-model prefill.

## Source

- Taekyung Heo et al., [*Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*](https://arxiv.org/abs/2608.03893), arXiv:2608.03893, 2026.

---

**Continue to:** [03-batching-scheduling](./03-batching-scheduling.md) for serving scheduling and cache management, or [04-memory-planning](./04-memory-planning.md) to budget receiver KV-cache capacity. Return to [index.json](../index.json) for the full chapter map.
