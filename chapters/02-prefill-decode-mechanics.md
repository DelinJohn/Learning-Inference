# Chapter 02 — Prefill and Decode Mechanics

> Why prefill is usually compute-bound, why decode is usually memory-bound, and how the KV cache changes the trade-off.

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

## Exact prefill and decode sequence

For a prompt containing $T$ tokens, each transformer layer calculates the prompt's queries, keys, and values largely in parallel:

$$
Q_{1:T},\qquad K_{1:T},\qquad V_{1:T}.
$$

That layer stores only its prompt keys and values in the KV cache. It does not cache the previous queries or the attention-score matrix.

The final prompt query reads the prompt cache:

$$
o_T
=
\operatorname{softmax}\left(
\frac{q_TK_{1:T}^{\mathsf T}}{\sqrt{d_{\text{head}}}}
\right)V_{1:T}.
$$

After the remaining layers and vocabulary projection, this produces the distribution used to choose the first generated token $x_{T+1}$.

That generated token is then passed through the model. At each layer, the model calculates

$$
q_{T+1},\qquad k_{T+1},\qquad v_{T+1}
$$

and appends the new key and value to that layer's cache:

$$
K_{1:T+1}=[K_{1:T};k_{T+1}],
\qquad
V_{1:T+1}=[V_{1:T};v_{T+1}].
$$

The new query reads the complete cache:

$$
o_{T+1}
=
\operatorname{softmax}\left(
\frac{q_{T+1}K_{1:T+1}^{\mathsf T}}{\sqrt{d_{\text{head}}}}
\right)V_{1:T+1}.
$$

This eventually predicts $x_{T+2}$, and the process repeats:

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

The timing distinction matters: the first generated token is predicted by the last prompt position. That generated token is then fed back into the model to predict the following token. The query performs the lookup; cached keys and values provide the memory being searched.

## How the model produces a name that is not one vocabulary token

A model does not require every complete word or name to have its own vocabulary entry. Before inference, the tokenizer splits text into representable pieces called tokens. Depending on the tokenizer, the name `Delin` might be represented illustratively as:

```text
" Del" + "in"
```

or:

```text
"D" + "el" + "in"
```

The exact split depends on the particular model's tokenizer, capitalization, leading whitespace, and normalization rules. Modern tokenizers commonly have subword and/or byte-level fallback pieces, allowing unusual names and strings to be represented even when the whole string is not one token.

For input text such as:

```text
My name is Delin.
```

the tokenizer converts every piece into a vocabulary token ID. The model receives the sequence of IDs and constructs contextual hidden states. Attention can use the prompt's name pieces when answering a later question such as `What is my name?`

Generation is autoregressive. Suppose the desired visible answer is `Delin` and the tokenizer uses two output tokens. The model performs:

```text
Step 1: LM-head logits → select token " Del"
Step 2: feed " Del" back into model → select token "in"
Detokenizer: " Del" + "in" → " Delin"
```

At each step, the vocabulary projection produces a logit only for existing vocabulary tokens:

$$
z_t=h_tW_{\text{vocab}}.
$$

It does not need a column for the complete name `Delin`; it needs columns for the smaller token pieces that compose it. The user sees complete text only after the detokenizer joins the generated pieces.

Token representability and factual knowledge are different questions. A tokenizer can represent a previously unseen name, but the model cannot know that it is the user's name unless the prompt, conversation state, retrieval system, fine-tuning, or some other available context provides that information.

## Why use a KV cache if reading it is expensive?

Without a KV cache, every decode step would recompute the keys and values of all previous tokens. That would repeatedly run transformer computations over the entire prefix.

The KV cache changes the trade-off:

- **Without caching:** repeat a large amount of computation at every step.
- **With caching:** compute each token's keys and values once, store them, and read them on later steps.

The cache therefore trades repeated computation for memory capacity and bandwidth. Reading cached values is much cheaper than recomputing the complete prefix, even though those reads contribute to the memory-bandwidth bottleneck.

---

**Continue to:** [03-batching-scheduling](./03-batching-scheduling.md) to see how batching changes the math above, or [04-memory-planning](./04-memory-planning.md) for the KV-cache sizing formula.
