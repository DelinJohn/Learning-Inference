# Chapter 11 — Training and Adaptation: LoRA, QLoRA, Distillation

> Companion notes for [09-architecture-evolution](./09-architecture-evolution.md). LoRA, QLoRA, distillation, and gradient averaging — what they each do and what they do not do.

## LoRA and QLoRA

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

## Distillation is separate from LoRA

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

## Gradient averaging and large batches

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

---

**End of notes.** Return to [index.json](../index.json) for the full chapter map and topic index.
