# Chapter 13 — JEPA: Predicting Embeddings and Understanding Stopgrad

## The core idea

**JEPA** means **Joint-Embedding Predictive Architecture**. It learns by predicting a target's representation from related context. An embedding is a learned vector of features, not a human-written description.

For example, hide a region of a dog photo. A pixel reconstruction model predicts that region's colors and textures. **I-JEPA**, the image version, instead predicts embeddings for the hidden region. Learning in feature space aims to emphasize useful structure without reconstructing every detail. JEPA is a learning framework; it is not itself a particular attention mechanism.

## Architecture and training

| Component | Input | Output |
|---|---|---|
| Context encoder | Visible image patches | Context embeddings |
| Predictor | Context embeddings and target positions | Predicted target embeddings |
| Target encoder | Full image | Embeddings selected at target positions |

I-JEPA uses Vision Transformer encoders and a smaller transformer predictor. Target positions tell the predictor where to predict; they do not reveal hidden pixels. Targets are selected from the full-image encoder output, rather than calculated by encoding each hidden crop independently.

```text
Visible patches → context encoder → predictor → predicted embeddings
                                      ↑                 │
                              target positions          │ compare → loss
                                                        │
Full image → target encoder → select target embeddings → stopgrad
```

Choose context and target masks, calculate both branches, compare their embeddings, then update the context encoder and predictor. No class labels are required. The trained encoder can subsequently supply features for downstream tasks.

Source: [I-JEPA paper](https://arxiv.org/abs/2301.08243).

## Reading the equations

Let I be the full image, x its visible context, and M the target positions:

$$
\hat z_y = p_\phi(f_\theta(x), M),
\qquad
z_y = \operatorname{stopgrad}\left([f_{\bar\theta}(I)]_M\right).
$$

- f is an encoder; p is the predictor.
- θ is the context encoder's parameters; φ is the predictor's parameters.
- θ̄ is the target encoder's parameters.
- The hat in ẑ means a prediction; brackets with M select target positions.

A simple illustrative loss is:

$$
\mathcal L = \frac{1}{N}\sum_{i=1}^{N}\|\hat z_{y,i}-z_{y,i}\|_2^2.
$$

This squared-error formula explains the gradient flow. The official I-JEPA implementation uses smooth L1 loss on normalized target features.

## What is stopgrad?

**Stopgrad means “stop the gradient.”** It preserves a value during the forward pass, but blocks backpropagation through that value's computation.

Its automatic-differentiation rule is:

$$
\operatorname{stopgrad}(u)=u \quad\text{in the forward pass},
\qquad
\frac{\partial\operatorname{stopgrad}(u)}{\partial u}=0
\quad\text{during backpropagation}.
$$

The zero derivative is a rule imposed on the computation graph, not the ordinary mathematical derivative of the identity function.

For a scalar example, suppose prediction a = 2 and target b = 5:

$$
L=(a-\operatorname{stopgrad}(b))^2=9.
$$

Backpropagation gives:

$$
\frac{\partial L}{\partial a}=2(a-b)=-6,
\qquad
\frac{\partial L}{\partial b}=0.
$$

Gradient descent therefore pushes a upward toward 5. This loss does not push b toward a. Without stopgrad, the derivative with respect to b would be +6, so both sides could move toward each other.

In PyTorch, `detach()` provides this behavior:

```python
prediction = predictor(context_encoder(visible_patches), target_positions)
target = target_encoder(full_image)[:, target_positions].detach()
loss = ((prediction - target) ** 2).mean()
loss.backward()  # gradients flow through prediction, not through target
```

This is schematic code: real I-JEPA handles patch masks and normalization explicitly. If the whole target branch needs no gradients, compute it inside `with torch.no_grad():` to avoid building its backward graph. `detach()` shares storage with the original tensor; it does not copy the values. [PyTorch detach documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

## How can the target encoder learn without gradients?

The target encoder starts as a copy of the context encoder and follows it through an **exponential moving average (EMA)**:

$$
\bar\theta \leftarrow \tau\bar\theta+(1-\tau)\theta.
$$

For an illustrative τ = 0.99, each update retains 99% of the old target parameter and mixes in 1% of the updated context parameter. This update is separate from backpropagation. Stopgrad does not freeze the target encoder forever.

A constant embedding for every image would be **representation collapse**: an easy prediction with no useful information. I-JEPA's asymmetric predictor, target update, stopped target gradients, and masking design form its training setup. Stopgrad or EMA alone does not guarantee that arbitrary architectures avoid collapse. See the [official training implementation](https://github.com/facebookresearch/ijepa/blob/main/src/train.py).

---

Return to [index.json](../index.json) for the chapter map and topic index.
