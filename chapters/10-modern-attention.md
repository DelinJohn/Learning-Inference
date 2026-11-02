# Chapter 10 — Modern Attention: MHA to KDA

> Companion notes for [09-architecture-evolution](./09-architecture-evolution.md). These clarifications are intentionally selective rather than a transcript.

## The central attention idea

For the current token:

- The **query** describes what information it is looking for.
- A **key** describes what information a previous token contains.
- A **value** is the information returned when its key matches the query.

Given the hidden-state matrix \(H\), the model learns three projections:

$$
Q = HW_Q, \qquad K = HW_K, \qquad V = HW_V
$$

Scaled dot-product attention is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^{\mathsf T}}{\sqrt{d_{\text{head}}}}+M
\right)V
$$

Here:

- \(H\): token hidden states.
- \(W_Q,W_K,W_V\): learned query, key, and value projection matrices.
- \(d_{\text{head}}\): number of features in one attention head.
- \(M\): causal mask that prevents a token from reading future tokens.

The calculation has two conceptually separate stages:

```text
QKᵀ → determine how much attention each previous token receives
   ↓
multiply those attention weights by V → retrieve and mix information
```

During prompt processing, the queries, keys, and values for all prompt positions can be calculated in parallel under the causal mask. During cached generation, only the new token's query, key, and value are calculated. Previous keys and values are reused; previous queries are not needed again.

## Why attention is split into heads

The hidden state is projected and reshaped into several heads:

$$
[B,T,d_{\text{model}}]
\longrightarrow
[B,T,N_{\text{heads}},d_{\text{head}}]
$$

Each head creates its own attention distribution, so different heads can learn different relationships. The head outputs are concatenated and projected:

$$
\operatorname{MHA}(H)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_n)W_O
$$

The raw hidden vector is not merely chopped into unrelated pieces first. Learned projection matrices create the head-specific representations, and implementations normally calculate the projections using large fused matrix multiplications.

## Math comparison: fixed position encoding versus CoPE

Let the token-embedding matrix be

$$
X\in\mathbb{R}^{T\times d_{\text{model}}},
$$

where $T$ is the number of tokens and $d_{\text{model}}$ is the hidden-vector width.

### Absolute position encoding

An absolute position table is a matrix

$$
P\in\mathbb{R}^{T\times d_{\text{model}}}.
$$

Row $P_i$ represents position $i$. It is added to the token embeddings:

$$
H=X+P.
$$

For learned absolute positions, the rows of $P$ are trainable vectors. For sinusoidal positions, the entries are fixed:

$$
P_{\text{pos},2r}
=
\sin\left(\frac{\text{pos}}{10000^{2r/d_{\text{model}}}}\right),
$$

$$
P_{\text{pos},2r+1}
=
\cos\left(\frac{\text{pos}}{10000^{2r/d_{\text{model}}}}\right).
$$

This is the fixed position matrix idea: position 10 always receives the representation for position 10, regardless of the surrounding text.

### Rotary and relative position methods

RoPE does not usually add a position matrix directly to $X$. It rotates queries and keys according to their positions:

$$
q_i'=R(i)q_i,
\qquad
k_j'=R(j)k_j.
$$

The attention score becomes:

$$
q_i'^{\mathsf T}k_j'
=
q_i^{\mathsf T}R(j-i)k_j.
$$

The score therefore depends on the relative distance $j-i$.

A simpler relative-position method adds a learned bias directly to the score:

$$
a_{i,j}=q_i^{\mathsf T}k_j+b_{i-j}.
$$

Here, $b_{i-j}$ is selected only from the token-distance difference. It does not depend on the actual content of the tokens.

### CoPE does not use one fixed $P$ matrix

CoPE dynamically creates a contextual-position number for every query/key pair. The gate matrix is approximately:

$$
G_{i,j}=g_{i,j}=\sigma(q_i^{\mathsf T}k_j),
\qquad
G\in\mathbb{R}^{T\times T}.
$$

The contextual-position matrix is formed by cumulative sums:

$$
P^{\text{ctx}}_{i,j}=p_{i,j}=\sum_{m=j}^{i}g_{i,m}.
$$

This is a $T\times T$ matrix of distances for one head, not a fixed $T\times d_{\text{model}}$ table added once to the input. It changes with the query and with the content of the sequence.

CoPE has a learned position-embedding table $e[0],e[1],\ldots,e[p_{\max}]$. It uses $p_{i,j}$ to select or interpolate a position vector, then modifies the attention score:

$$
a_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}].
$$

Finally:

$$
\alpha_{i,:}=\operatorname{softmax}(a_{i,:}),
\qquad
o_i=\sum_j\alpha_{i,j}v_j.
$$

### Tiny numerical position example

For query position $i=4$, suppose a head produces these gates while scanning backward:

```text
token j:       1     2     3     4
gate g₄ⱼ:     0.0   1.0   0.0   0.0
```

The resulting contextual distances are approximately:

```text
token j:       1     2     3     4
p₄ⱼ:          1.0   1.0   0.0   0.0
```

The head has effectively decided that token 2 marks a meaningful boundary. Tokens 3 and 4 are in the current contextual region, while tokens 1 and 2 are one contextual step away. With ordinary relative position, all four tokens would simply be separated by their raw token distances.

## A slower explanation of what the position matrices do

The word or token matrix and the position information are two different things.

Suppose the sequence is:

```text
Alice slept . Bob ran
```

The tokenizer and embedding table first produce one vector per token:

$$
X=
\begin{bmatrix}
x_{\text{Alice}}\\
x_{\text{slept}}\\
x_{.}\\
x_{\text{Bob}}\\
x_{\text{ran}}
\end{bmatrix}.
$$

Each row contains learned semantic features. At this stage, two occurrences of the same token receive the same embedding, and the matrix does not by itself say which occurrence came first.

### Absolute position changes the token vectors before attention

Absolute position encoding creates another matrix with the same shape:

$$
P=
\begin{bmatrix}
p_1\\p_2\\p_3\\p_4\\p_5
\end{bmatrix}.
$$

The model performs element-wise addition:

$$
H=X+P.
$$

For the fourth token:

$$
h_{\text{Bob}}=x_{\text{Bob}}+p_4.
$$

The position vector $p_4$ does not replace the meaning of `Bob`. It slightly changes every feature in the `Bob` embedding so subsequent layers can distinguish "Bob at position 4" from "Bob at position 1."

After this addition, the model creates queries, keys, and values:

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

Thus, with absolute position encoding, position affects attention indirectly because position was mixed into $H$ before $Q$, $K$, and $V$ were produced.

### Relative position changes the attention-score matrix

Attention first creates a content-score matrix:

$$
S_{\text{content}}=\frac{QK^{\mathsf T}}{\sqrt{d_{\text{head}}}}.
$$

Rows represent queries and columns represent keys:

```text
                 key token
             1      2      3
query 1     S₁₁    blocked blocked
query 2     S₂₁    S₂₂    blocked
query 3     S₃₁    S₃₂    S₃₃
```

For a small numerical example, suppose:

$$
S_{\text{content}}=
\begin{bmatrix}
2&-\infty&-\infty\\
1&2&-\infty\\
0.5&1&2
\end{bmatrix}.
$$

These numbers say how well the token contents match. Relative position then adds a distance bias:

$$
B=
\begin{bmatrix}
0&-\infty&-\infty\\
-0.5&0&-\infty\\
-1&-0.5&0
\end{bmatrix}.
$$

Adding the two gives:

$$
S_{\text{final}}
=
S_{\text{content}}+B
=
\begin{bmatrix}
2&-\infty&-\infty\\
0.5&2&-\infty\\
-0.5&0.5&2
\end{bmatrix}.
$$

For query 3, the raw content scores were $[0.5,1,2]$. After applying distance penalties, they became $[-0.5,0.5,2]$. Softmax converts these into approximately:

$$
\operatorname{softmax}([-0.5,0.5,2])
\approx
[0.063,0.171,0.766].
$$

In this chosen example, the oldest key still participates but its score is reduced because it is farther away. A learned relative bias does not have to be negative; training can learn to favor a particular distance. What remains fixed is that the same bias for a given distance is used regardless of whether the token is a name, punctuation mark, or code keyword.

### Relative position from distance matrix to final output

Use three tokens:

```text
position 1: I
position 2: love
position 3: cats
```

For causal attention, define relative distance as:

$$
d_{i,j}=i-j,
$$

where $i$ is the query position and $j$ is the key position. The distance matrix is:

$$
D=
\begin{bmatrix}
0&\text{blocked}&\text{blocked}\\
1&0&\text{blocked}\\
2&1&0
\end{bmatrix}.
$$

Read the final row in words:

```text
Query "cats" to key "I"    → distance 2
Query "cats" to key "love" → distance 1
Query "cats" to key "cats" → distance 0
```

The model keeps a learned lookup from distance to score bias. For this toy example, choose:

$$
b_0=0,
\qquad
b_1=-0.5,
\qquad
b_2=-1.
$$

Replace every distance in $D$ with its corresponding bias:

$$
B=
\begin{bmatrix}
b_0&-\infty&-\infty\\
b_1&b_0&-\infty\\
b_2&b_1&b_0
\end{bmatrix}
=
\begin{bmatrix}
0&-\infty&-\infty\\
-0.5&0&-\infty\\
-1&-0.5&0
\end{bmatrix}.
$$

Now deliberately make all allowed content scores equal so that only the positional effect is visible:

$$
S_{\text{content}}=
\begin{bmatrix}
1&-\infty&-\infty\\
1&1&-\infty\\
1&1&1
\end{bmatrix}.
$$

Add the bias matrix:

$$
S_{\text{final}}
=
S_{\text{content}}+B
=
\begin{bmatrix}
1&-\infty&-\infty\\
0.5&1&-\infty\\
0&0.5&1
\end{bmatrix}.
$$

For the query `cats`, use the final row:

$$
[0,0.5,1].
$$

Softmax produces:

$$
\operatorname{softmax}([0,0.5,1])
\approx
[0.186,0.307,0.507].
$$

So `cats` reads approximately:

```text
18.6% from the value of "I"
30.7% from the value of "love"
50.7% from its own value
```

To finish the attention operation, let the value vectors be:

$$
V=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
$$

The output for `cats` is the weighted sum:

$$
o_3
=
0.186[1,0]
+
0.307[0,1]
+
0.507[1,1]
$$

$$
o_3\approx[0.693,0.814].
$$

Without the relative-position bias, the three equal content scores would produce equal attention weights of one third. The bias changes those weights according to relative distance before the values are mixed.

Relative position is also shift-invariant. The sequences at absolute positions $[1,2,3]$ and $[101,102,103]$ produce the same distance pattern:

$$
(3-1,3-2,3-3)=(2,1,0),
$$

$$
(103-101,103-102,103-103)=(2,1,0).
$$

That is why it is called relative position: the calculation cares about the distance between query and key, not their absolute sequence numbers.

There are several implementations of this idea:

- A learned scalar bias uses $a_{i,j}=q_i^{\mathsf T}k_j+b_{i-j}$.
- A relative vector uses $a_{i,j}=q_i^{\mathsf T}(k_j+r_{i-j})$.
- ALiBi uses a fixed head-specific linear penalty such as $-m_h(i-j)$.
- RoPE rotates queries and keys so their dot product depends on $i-j$.

All four make attention depend on relative position, but they inject that information at different points in the calculation.

### Why a learned relative bias is updated

The values $b_0,b_1,b_2,\ldots$ can be trainable model parameters, just like entries in $W_Q$ or $W_K$. They are often initialized to zero or small values. Because they are added to attention logits, they affect the complete prediction path:

```text
relative bias bᵣ
    ↓
attention logits
    ↓
softmax probabilities
    ↓
weighted value mixture
    ↓
later transformer layers
    ↓
next-token probabilities
    ↓
training loss
```

Backpropagation therefore computes how changing each bias would change the loss. If $r$ denotes a relative distance, then:

$$
\frac{\partial \mathcal L}{\partial b_r}
=
\sum_{i,j:\,i-j=r}
\frac{\partial \mathcal L}{\partial a_{i,j}}.
$$

Every query/key pair at distance $r$ contributes to the same bias gradient. The optimizer then performs an update conceptually like:

$$
b_r
\leftarrow
b_r-\eta
\frac{\partial\mathcal L}{\partial b_r},
$$

where $\eta$ is the learning rate.

For one query, suppose the gradient arriving at three attention logits is:

$$
\frac{\partial\mathcal L}{\partial a_{3,:}}
=
[0.2,-0.4,0.2].
$$

Those three logits correspond to distances $[2,1,0]$, so this example contributes:

$$
\frac{\partial\mathcal L}{\partial b_2}=0.2,
\qquad
\frac{\partial\mathcal L}{\partial b_1}=-0.4,
\qquad
\frac{\partial\mathcal L}{\partial b_0}=0.2.
$$

For distance 1:

$$
b_1\leftarrow b_1-\eta(-0.4)=b_1+0.4\eta.
$$

Thus this training example pushes $b_1$ upward, making attention to keys one token behind slightly easier in a future forward pass. Other examples may push it downward. The accumulated training data determines the final compromise.

For example, language often contains useful adjacent relationships such as an adjective followed by a noun or a first name followed by a surname. If attending one token backward repeatedly improves next-token predictions, training can increase the corresponding bias. Other heads or layers may learn different patterns, including preferences for longer distances.

The bias is updated only during training or fine-tuning. During ordinary inference, it is frozen and simply read as part of the model weights. Depending on the architecture, biases may be separate per layer and attention head, and large distances may be grouped into buckets rather than receiving an individual parameter.

### RoPE changes query and key directions before their dot product

RoPE does not usually add a bias after $QK^{\mathsf T}$. Instead, it rotates each query and key based on its position.

Take two positions and deliberately simple two-dimensional vectors:

$$
q_2=[1,0],\qquad k_1=[1,0],\qquad k_2=[1,0].
$$

Without position information, both dot products are identical:

$$
q_2^{\mathsf T}k_1=1,
\qquad
q_2^{\mathsf T}k_2=1.
$$

Use no rotation at position 1 and a 90-degree rotation at position 2:

$$
R_1=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
R_2=
\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
$$

After RoPE:

$$
q_2'=R_2q_2=[0,1],
$$

$$
k_1'=R_1k_1=[1,0],
\qquad
k_2'=R_2k_2=[0,1].
$$

The new scores are:

$$
q_2'^{\mathsf T}k_1'=0,
\qquad
q_2'^{\mathsf T}k_2'=1.
$$

The contents were deliberately identical, but their relative positions changed the dot products. Real RoPE uses much smaller angles at several frequencies and rotates every pair of features inside each head.

#### RoPE from the actual formula

RoPE means **Rotary Position Embedding**. Start with one pair of features from a query or key vector:

$$
x=
\begin{bmatrix}
x_0\\x_1
\end{bmatrix}.
$$

For token position $m$ and angular frequency $\theta$, RoPE uses the rotation matrix:

$$
R(m\theta)=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}.
$$

The rotated pair is:

$$
\widetilde x=R(m\theta)x.
$$

Writing out both coordinates:

$$
\widetilde x_0
=
x_0\cos(m\theta)-x_1\sin(m\theta),
$$

$$
\widetilde x_1
=
x_0\sin(m\theta)+x_1\cos(m\theta).
$$

The vector length does not change. Only its direction changes:

$$
\|\widetilde x\|_2=\|x\|_2.
$$

Therefore RoPE injects position without making the query or key larger merely because it occurs later in the sequence.

#### A head uses many rotation speeds

If the head dimension is $d_{\text{head}}$, adjacent features are grouped into pairs:

```text
(feature 0, feature 1) → rotation pair 0
(feature 2, feature 3) → rotation pair 1
(feature 4, feature 5) → rotation pair 2
...
```

The feature-pair index is not the same as the query/key token indices $i$ and $j$. For one attention head with three tokens and head dimension four, the query matrix has the shape:

$$
Q=
\begin{bmatrix}
q_{0,0}&q_{0,1}&q_{0,2}&q_{0,3}\\
q_{1,0}&q_{1,1}&q_{1,2}&q_{1,3}\\
q_{2,0}&q_{2,1}&q_{2,2}&q_{2,3}
\end{bmatrix}.
$$

The first subscript selects a token row. The second selects a feature column:

```text
i or j → token position / matrix row
c      → feature coordinate / matrix column
r      → pair number formed from two feature columns
```

For query token $i=2$:

$$
q_2=
\begin{bmatrix}
q_{2,0}&q_{2,1}&q_{2,2}&q_{2,3}
\end{bmatrix}.
$$

RoPE forms two feature pairs inside that one row:

$$
q_2^{(0)}=[q_{2,0},q_{2,1}],
\qquad
q_2^{(1)}=[q_{2,2},q_{2,3}].
$$

The same pairing is applied inside every key row. For key token $j=1$:

$$
k_1^{(0)}=[k_{1,0},k_{1,1}],
\qquad
k_1^{(1)}=[k_{1,2},k_{1,3}].
$$

RoPE rotates query pair $r$ by $i\theta_r$ and key pair $r$ by $j\theta_r$. Their complete attention score sums the dot products from all feature pairs:

$$
\widetilde q_i^{\mathsf T}\widetilde k_j
=
\sum_r
\left(R(i\theta_r)q_i^{(r)}\right)^{\mathsf T}
\left(R(j\theta_r)k_j^{(r)}\right).
$$

Thus $i$ and $j$ answer "which two tokens are being compared?" while $r$ answers "which two coordinates inside their query/key vectors are currently being rotated?"

Each pair uses a different frequency. A common base formulation is:

$$
\theta_r
=
10000^{-2r/d_{\text{head}}},
\qquad
r=0,1,\ldots,\frac{d_{\text{head}}}{2}-1.
$$

For a four-dimensional toy head:

$$
\theta_0=1,
\qquad
\theta_1=0.01.
$$

Here, $\theta_r$ is not a hidden-state value. It is the fixed rotation frequency assigned to feature pair $r$:

```text
x₀, x₁, x₂, x₃ → query/key feature values derived from the hidden state
r                 → feature-pair number
θᵣ                → rotation per token for pair r
m                 → token position
mθᵣ               → actual angle used at position m
```

For example, if $\theta_1=0.01$ radians per token and $m=20$, that pair rotates by:

$$
m\theta_1=20\times0.01=0.2\text{ radians}.
$$

The values being rotated might be $[x_2,x_3]=[2.4,-0.7]$. Those values come from the query or key projection; $\theta_1$ only tells RoPE how far to rotate them.

Model configuration files can make the terminology confusing. A field commonly called `rope_theta` is often the base $B$, such as $B=10000$, rather than one hidden value or one final angle. The per-pair frequencies are generated from that base:

$$
\theta_r=B^{-2r/d_{\text{head}}},
$$

and the angle for a particular token is still $m\theta_r$.

#### What determines each RoPE frequency

In standard RoPE, four quantities determine the rotation behavior:

1. The architecture chooses a base $B$, commonly exposed as `rope_theta`.
2. The architecture chooses how many query/key dimensions receive RoPE.
3. The feature-pair index $r$ selects one frequency from the geometric sequence.
4. The token position $m$ multiplies that frequency to produce the final angle.

If all head dimensions are rotated, the frequency is:

$$
\theta_r=B^{-2r/d_{\text{head}}}.
$$

If only $d_{\text{rotary}}$ dimensions are rotated, implementations commonly use that rotary dimension in the denominator instead:

$$
\theta_r=B^{-2r/d_{\text{rotary}}}.
$$

For $B=10000$ and $d_{\text{rotary}}=8$, there are four feature pairs:

| Pair $r$ | Dimensions | Frequency $\theta_r$ in radians/token | Approximate tokens for one full turn $2\pi/\theta_r$ |
|---:|---|---:|---:|
| 0 | 0 and 1 | $1$ | $6.28$ |
| 1 | 2 and 3 | $0.1$ | $62.8$ |
| 2 | 4 and 5 | $0.01$ | $628$ |
| 3 | 6 and 7 | $0.001$ | $6{,}283$ |

This geometric spacing is intentional. The fast pair changes phase strongly over nearby tokens, while slow pairs preserve phase information across hundreds or thousands of tokens. The model uses all pairs together, so it does not identify a distance from only one repeating clock.

Changing each input has a different effect:

```text
Increase base B
    → most pairs rotate more slowly
    → longer wavelengths

Increase rotary dimension
    → more frequency pairs
    → denser coverage of short-to-long distance scales

Increase pair index r
    → lower frequency for that pair
    → slower rotation

Increase token position m
    → advance the phase mθᵣ
    → frequency θᵣ itself does not change
```

Standard RoPE frequencies are architectural constants, not values selected from the hidden state and not normally updated by backpropagation. Long-context RoPE variants can rescale the position $m$, the frequencies $\theta_r$, or both; that changes effective rotation speeds but is still a configured positional rule.

#### Who determines the RoPE base, and is it trained?

In standard RoPE, the model designer chooses the base $B$ before pretraining. It is a hyperparameter in the model configuration, not a parameter updated by the optimizer:

```text
Before training:
    choose rope_theta/base B

During training:
    keep B fixed
    update WQ, WK, and the other model weights

During inference:
    read the same B from the model configuration
    generate the required sine and cosine values
```

There is no unique formula that derives the correct base from parameter count. Designers choose it using the intended training context, rotary dimension, desired range of wavelengths, training stability, and empirical evaluation on short- and long-context tasks. The common value $10000$ comes from the geometric-frequency design used by sinusoidal position encodings; it is a convention and useful starting point, not a universal optimum.

The slowest frequency is approximately the frequency of the final pair:

$$
\theta_{\min}
=
B^{-2r_{\max}/d_{\text{rotary}}},
\qquad
r_{\max}=\frac{d_{\text{rotary}}}{2}-1.
$$

Its wavelength is:

$$
\lambda_{\max}=\frac{2\pi}{\theta_{\min}}.
$$

Increasing $B$ lowers most frequencies and increases their wavelengths. The fastest pair remains $\theta_0=1$ because its exponent is zero. Thus a larger base mainly stretches the medium and slow rotation pairs rather than uniformly slowing every pair.

Although $B$ is fixed, the model still learns how to use RoPE. The trainable projections $W_Q$ and $W_K$ learn query/key features whose rotated dot products are useful for prediction. Changing $B$ after training changes those dot products, so an arbitrary post-training change can damage accuracy. Long-context scaling methods make controlled changes and are normally validated or accompanied by continued training.

Some research architectures explicitly make frequencies or scaling factors trainable. In that case they receive gradients because the architecture defines them as parameters. That is an exception; the presence of a `rope_theta` number in a normal model configuration does not mean it was learned.

At position $m=2$, the first pair rotates by $2$ radians while the second rotates by only $0.02$ radians:

$$
\widetilde x=
\begin{bmatrix}
x_0\cos 2-x_1\sin 2\\
x_0\sin 2+x_1\cos 2\\
x_2\cos 0.02-x_3\sin 0.02\\
x_2\sin 0.02+x_3\cos 0.02
\end{bmatrix}.
$$

Fast rotation pairs change substantially over nearby tokens. Slow pairs retain useful phase relationships over longer distances. Combining several frequencies gives the head a richer representation of relative displacement than one rotation speed would provide.

#### Why the dot product contains relative position

Let query $q_m$ occur at position $m$ and key $k_n$ at position $n$. RoPE rotates both:

$$
\widetilde q_m=R_mq_m,
\qquad
\widetilde k_n=R_nk_n.
$$

Their attention score is:

$$
\widetilde q_m^{\mathsf T}\widetilde k_n
=
(R_mq_m)^{\mathsf T}(R_nk_n).
$$

Move the rotations around the dot product:

$$
\widetilde q_m^{\mathsf T}\widetilde k_n
=
q_m^{\mathsf T}R_m^{\mathsf T}R_nk_n.
$$

For rotation matrices:

$$
R_m^{\mathsf T}=R_{-m},
$$

so:

$$
R_m^{\mathsf T}R_n=R_{n-m}.
$$

Therefore:

$$
\boxed{
\widetilde q_m^{\mathsf T}\widetilde k_n
=
q_m^{\mathsf T}R_{n-m}k_n
}
$$

The absolute positions $m$ and $n$ collapse into the difference $n-m$. That is the mathematical reason RoPE gives the dot product relative-position information.

#### Three-token RoPE matrix

Use zero-based positions $0,1,2$ and a deliberately exaggerated 90-degree rotation per step. Suppose the unrotated query and key at every position are identical:

$$
Q=K=
\begin{bmatrix}
1&0\\
1&0\\
1&0
\end{bmatrix}.
$$

Without position information:

$$
QK^{\mathsf T}=
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}.
$$

Every content score is identical. Apply the position rotations:

$$
R_0=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
R_1=
\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\quad
R_2=
\begin{bmatrix}-1&0\\0&-1\end{bmatrix}.
$$

The rotated queries and keys are:

$$
\widetilde Q=\widetilde K=
\begin{bmatrix}
1&0\\
0&1\\
-1&0
\end{bmatrix}.
$$

Now:

$$
\widetilde Q\widetilde K^{\mathsf T}
=
\begin{bmatrix}
1&0&-1\\
0&1&0\\
-1&0&1
\end{bmatrix}.
$$

After applying the causal mask:

$$
S=
\begin{bmatrix}
1&-\infty&-\infty\\
0&1&-\infty\\
-1&0&1
\end{bmatrix}.
$$

For the third query, softmax sees:

$$
[-1,0,1]
$$

and produces approximately:

$$
\operatorname{softmax}([-1,0,1])
\approx
[0.090,0.245,0.665].
$$

All three unrotated keys were identical. RoPE made their attention scores different only because their positions relative to the query were different.

#### Where RoPE occurs in an LLM layer

The usual order is:

```text
Hidden states H
    ↓
Q = HWQ, K = HWK, V = HWV
    ↓
reshape Q, K, V into attention heads
    ↓
apply position-specific rotations to Q and K
    ↓
calculate rotated-Q × rotated-Kᵀ
    ↓
softmax
    ↓
multiply by V
```

Values are normally not rotated. They are the information returned after the rotated query/key match determines the attention weights.

#### RoPE during prefill and cached decode

During prefill, the model computes the sine and cosine values for all prompt positions and rotates all prompt queries and keys in parallel. The rotated keys are then stored in the layer's KV cache together with the values.

During decode at new position $m$:

```text
1. Compute qₘ, kₘ, and vₘ for the new token.
2. Rotate qₘ and kₘ using position m.
3. Compare rotated qₘ with the cached, already-rotated keys.
4. Append rotated kₘ and vₘ to the cache.
5. Use the attention weights to mix the cached values.
```

RoPE does not add another per-token object comparable in size to the KV cache. Implementations can precompute or efficiently generate sine/cosine tables. The important serving requirement is that every request uses the correct logical position when its new query and key are rotated.

#### Is RoPE learned?

In standard RoPE, the sine/cosine frequencies are fixed rather than trained as a bias table. The query and key projection weights $W_Q$ and $W_K$ are trained, so the model learns feature representations that work well under those rotations. Some later variants modify or scale the frequencies, especially for longer contexts.

The formula can be evaluated at positions longer than those seen during training, but that does not guarantee good model behavior. At much longer lengths, the model sees unfamiliar rotation phases and some dimensions may rotate too quickly. Long-context techniques such as RoPE scaling, NTK-aware scaling, and YaRN adjust the frequencies or position mapping to reduce this extrapolation problem.

### CoPE first decides what should count as distance

Now return to:

```text
1 Alice   2 slept   3 .   4 Bob   5 ran
```

Suppose query 5 belongs to an attention head that has learned to count sentence boundaries. For this query, the gates might be:

$$
g_{5,:}=
\begin{bmatrix}
0&0&1&0&0
\end{bmatrix}.
$$

The gate is 1 at the period and 0 elsewhere. Starting at the query and accumulating backward gives:

$$
p_{5,:}=
\begin{bmatrix}
1&1&1&0&0
\end{bmatrix}.
$$

This means:

```text
Alice, slept, . → one sentence boundary away
Bob, ran        → current sentence region
```

CoPE then converts contextual positions 0 and 1 into learned vectors. For a two-dimensional toy example, suppose:

$$
e[0]=[0.2,0],
\qquad
e[1]=[1,0],
\qquad
q_5=[1,0].
$$

The position bonuses are:

$$
q_5^{\mathsf T}e[1]=1,
\qquad
q_5^{\mathsf T}e[0]=0.2.
$$

Therefore the CoPE position contribution for all five keys is:

$$
S_{\text{CoPE-position}}
=
\begin{bmatrix}
1&1&1&0.2&0.2
\end{bmatrix}.
$$

Suppose the ordinary content scores for query 5 are:

$$
S_{\text{content}}
=
\begin{bmatrix}
0.3&0.6&0.1&0.4&0.2
\end{bmatrix}.
$$

CoPE adds content and contextual-position scores:

$$
S_{\text{final}}
=
\begin{bmatrix}
0.3&0.6&0.1&0.4&0.2
\end{bmatrix}
+
\begin{bmatrix}
1&1&1&0.2&0.2
\end{bmatrix}
$$

$$
S_{\text{final}}
=
\begin{bmatrix}
1.3&1.6&1.1&0.6&0.4
\end{bmatrix}.
$$

In this toy head, the learned query is looking for information in the previous sentence, so tokens at contextual position 1 receive the larger position bonus. Another query or another head can generate different gates and prefer contextual position 0 instead.

Nothing in CoPE hard-codes "period means sentence boundary." Training discovers which keys should open the gate and what contextual positions each query should prefer.

### What is actually computed during prefill and decode

During prefill, every query has its own gate row:

$$
G=
\begin{bmatrix}
g_{1,1}&0&0&\cdots\\
g_{2,1}&g_{2,2}&0&\cdots\\
g_{3,1}&g_{3,2}&g_{3,3}&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{bmatrix}.
$$

Each row is cumulatively summed to produce contextual positions for that query. During cached decoding, old rows do not have to be recalculated. The new query computes one new gate row against the cached keys, obtains its contextual positions, and uses them to form its attention scores.

So the central difference is:

```text
Absolute PE → change token vectors using fixed position rows
Relative PE → add a fixed token-distance bias to the score matrix
RoPE        → rotate Q and K using fixed token positions
CoPE        → use Q and K to decide dynamically what counts as distance
```

## Matrix examples: absolute position, relative position, RoPE, and CoPE

Use three tokens and a hidden size of four. Let the token embeddings be:

$$
X=
\begin{bmatrix}
1&2&0&1\\
0&1&1&2\\
2&0&1&1
\end{bmatrix}.
$$

### Absolute learned position

A learned position table could be:

$$
P=
\begin{bmatrix}
0.1&0.1&0.2&0.2\\
0.2&0.2&0.3&0.3\\
0.3&0.3&0.4&0.4
\end{bmatrix}.
$$

The transformer input becomes:

$$
H=X+P=
\begin{bmatrix}
1.1&2.1&0.2&1.2\\
0.2&1.2&1.3&2.3\\
2.3&0.3&1.4&1.4
\end{bmatrix}.
$$

The important point is that $P$ is one fixed $T\times d_{\text{model}}$ matrix. Position 2 receives the same row whenever it appears in a sequence of this length.

### Sinusoidal position

Sinusoidal encoding also creates a position matrix, but its entries are generated rather than learned:

$$
P_{\text{pos},2r}
=
\sin\left(\frac{\text{pos}}{10000^{2r/d}}\right),
\qquad
P_{\text{pos},2r+1}
=
\cos\left(\frac{\text{pos}}{10000^{2r/d}}\right).
$$

For $d=4$, its first rows are approximately:

$$
P\approx
\begin{bmatrix}
0&1&0&1\\
0.8415&0.5403&0.0100&0.99995\\
0.9093&-0.4161&0.0200&0.99980
\end{bmatrix}.
$$

It is still fixed by token index; it does not inspect the token content before assigning position.

### Relative-position bias

Relative position often leaves $H$ unchanged and adds a bias to the attention logits. For a causal sequence of three tokens, choose:

$$
b_0=0,\qquad b_1=-0.5,\qquad b_2=-1.0.
$$

The bias matrix added to $QK^{\mathsf T}$ is:

$$
B=
\begin{bmatrix}
0&-\infty&-\infty\\
-0.5&0&-\infty\\
-1.0&-0.5&0
\end{bmatrix}.
$$

The negative infinity entries are the causal mask. The score calculation is:

$$
A=\frac{QK^{\mathsf T}}{\sqrt{d_{\text{head}}}}+B.
$$

The value $b_1$ is used whenever a key is one token behind the query, regardless of what that token means.

### RoPE matrix example

For a deliberately tiny two-dimensional head, use a 90-degree rotation per position:

$$
R(0)=
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
R(1)=
\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\quad
R(2)=
\begin{bmatrix}-1&0\\0&-1\end{bmatrix}.
$$

If the unrotated query and key at every position are $[1,0]$, the rotated vectors are:

$$
Q'=K'=
\begin{bmatrix}
1&0\\
0&1\\
-1&0
\end{bmatrix}.
$$

Their pairwise dot products are:

$$
Q'K'^{\mathsf T}=
\begin{bmatrix}
1&0&-1\\
0&1&0\\
-1&0&1
\end{bmatrix}.
$$

The score between positions 0 and 1 is 0, while the score between positions 0 and 2 is $-1$. The change comes from their relative rotation. Real RoPE uses many two-dimensional rotation pairs inside a larger head; this small example only exposes the idea.

### CoPE matrix example

CoPE first computes a gate matrix. To isolate the position effect, suppose a three-token head has learned:

$$
G=
\begin{bmatrix}
0&0&0\\
0.6&0&0\\
0.2&0.8&0
\end{bmatrix}.
$$

The entries are contextual gates, not fixed token distances. Cumulative sums produce:

$$
P^{\text{ctx}}=
\begin{bmatrix}
0&0&0\\
0.6&0&0\\
1.0&0.8&0
\end{bmatrix}.
$$

For example, for query 3:

$$
p_{3,1}=0.2+0.8=1.0,
\qquad
p_{3,2}=0.8,
\qquad
p_{3,3}=0.
$$

Suppose the learned position vectors are $e[0]=[0,0]$, $e[1]=[1,0]$, and $e[2]=[2,0]$. Then:

$$
e[0.8]=0.2e[0]+0.8e[1]=[0.8,0].
$$

If the query is $q_3=[1,0]$, the positional contributions to its three logits are:

$$
q_3^{\mathsf T}e[1.0]=1.0,
\qquad
q_3^{\mathsf T}e[0.8]=0.8,
\qquad
q_3^{\mathsf T}e[0]=0.
$$

The real attention logits add these contributions to the ordinary content scores:

$$
a_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}].
$$

This is the essential matrix difference:

```text
Absolute PE:  one fixed T × d_model matrix P
Relative PE:  one fixed distance-bias matrix B added to logits
RoPE:         rotate Q and K using their fixed token positions
CoPE:         compute a query/content-dependent T × T gate and position matrix
```

CoPE's $P^{\text{ctx}}$ is usually computed in tiles or implicitly; it does not mean that an enormous permanent matrix must be stored in the KV cache.

## Contextual Position Encoding

**Contextual Position Encoding (CoPE)** is still a type of position encoding. Ordinary absolute, relative, and rotary position methods measure distance using token positions. CoPE allows an attention head to learn which tokens should advance its position counter.

For query position $i$ and earlier token $j$, a simplified gate is

$$
g_{i,j}=\sigma(q_i^{\mathsf T}k_j).
$$

The contextual distance is obtained by accumulating gates rather than blindly counting every token:

$$
p_{i,j}=\sum_{m=j}^{i}g_{i,m}.
$$

Here:

- $g_{i,j}$: how much token $j$ should count as a positional step for query $i$;
- $p_{i,j}$: learned contextual position of token $j$ relative to query $i$;
- $\sigma$: sigmoid function, producing a value between zero and one.

The contextual position changes the attention score:

$$
a_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}].
$$

The scores are then passed through softmax and used to mix values as usual. Fractional contextual positions are handled by interpolating nearby learned position embeddings.

For example, ordinary relative position counts every token. A CoPE head can instead learn to count only sentence endings:

```text
current sentence  → contextual position 0
previous sentence → contextual position 1
sentence before   → contextual position 2
```

Another head could count words, code statements, paragraph boundaries, or another learned feature. The meaning of position can therefore depend on both the current query and preceding content.

CoPE is part of the attention-score calculation, but it does not replace query/key/value attention, reduce the KV cache, or change full attention's sequence-length scaling. It is learned during training, not an inference option that can normally be enabled on an existing model. The [original CoPE paper](https://arxiv.org/abs/2405.18719) evaluated relatively small models and explicitly did not establish results for billion-parameter language models, so it should not be assumed to be a standard feature of deployed LLMs.

### CoPE step by step

Suppose the sequence is:

```text
Alice wrote code .  Bob reviewed it .
```

With ordinary relative position, the model counts every token. The word `code` might be five token positions away from the current query, even though it is only one sentence away.

CoPE gives each attention head a learned counting rule:

```text
Head 1: count every token
Head 2: count word beginnings
Head 3: count sentence-ending tokens
```

For a query at the final period, the sentence-counting head can assign:

```text
tokens in the current sentence → contextual distance 0
tokens in the previous sentence → contextual distance 1
```

The operation inside one attention head is:

```text
1. Produce qᵢ and all previous kⱼ vectors.
2. Compute a gate gᵢⱼ for each previous token.
3. Accumulate the gates from the current query backwards.
4. Use the resulting contextual distance pᵢⱼ to obtain a position embedding.
5. Add that positional contribution to the content-based attention score.
6. Apply softmax and use the weights to mix the value vectors.
```

The gate is soft rather than a hard yes/no decision. For example:

$$
g_{i,j}=0.05
$$

means that token $j$ contributes very little to the position counter, while $g_{i,j}=0.95$ contributes almost one position step. Because gates are fractional, the contextual distance can also be fractional. If

$$
p_{i,j}=1.6,
$$

the model interpolates between the learned position embeddings for positions 1 and 2:

$$
e[1.6]
=
0.4e[1]+0.6e[2].
$$

The final score can be written in either of two equivalent forms:

$$
a_{i,j}
=
q_i^{\mathsf T}(k_j+e[p_{i,j}])
$$

or, after distributing the multiplication,

$$
a_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}].
$$

The first term asks, "Does this token contain the information I need?" The second asks, "Does this token have the contextual distance or type of position I need?"

### How CoPE differs from ordinary positional methods

```text
Absolute position:  token 37 has position 37
Relative position:  token is 5 tokens behind the query
RoPE:               encode token distance through rotations of q and k
CoPE:               count context-selected events, possibly differently per head
```

CoPE can be combined with a normal relative or rotary method. It is therefore better understood as an additional learned way to address positions, not as a replacement for the entire attention mechanism.

### CoPE and inference engineering

CoPE does not make attention linear. A query may still compare with every previous key, and the KV cache still grows with the number of cached tokens. It adds gates, cumulative-position calculations, and position-score work around the usual attention calculation. The paper describes ways to reuse the query/key products and limit the maximum contextual position, but an optimized CoPE kernel is still required for its theoretical benefits to become practical.

Consequently, an inference engineer should treat CoPE as an architecture property:

```text
Existing RoPE model → cannot normally switch on CoPE at serving time
Model trained with CoPE → serving engine must support its extra score computation
```

The main potential benefit is better context-dependent retrieval, such as finding the previous sentence or the previous relevant code statement. The main cost is additional attention computation and less mature deployment support.

### Complete CoPE reference: shapes, efficient algorithm, training, variants, and limits

#### Where CoPE sits in an attention layer

For one attention head with sequence length $T$ and head dimension $d_h$:

$$
Q,K,V\in\mathbb{R}^{T\times d_h}.
$$

Ordinary causal attention forms content logits:

$$
L
=
\frac{QK^{\mathsf T}}{\sqrt{d_h}}+M,
\qquad
L\in\mathbb{R}^{T\times T},
$$

where $M$ is the causal mask. CoPE operates after these query/key content logits exist and before the final softmax:

```text
Q, K
  ↓
content logits L = QKᵀ/√dₕ + causal mask
  ↓
CoPE gates and contextual positions
  ↓
CoPE position-logit contribution
  ↓
final logits
  ↓
softmax
  ↓
weighted sum of V
```

CoPE does not replace $Q$, $K$, $V$, the causal mask, softmax, or the value mixture.

#### Gate matrix

The paper's simplified mathematical gate is:

$$
g_{i,j}=\sigma(q_i^{\mathsf T}k_j),
\qquad j<i.
$$

Across the sequence, these gates form a causal matrix:

$$
G\in\mathbb{R}^{T\times T}.
$$

Row $i$ belongs to query token $i$; column $j$ belongs to key token $j$. Each entry answers:

> For this query and this attention head, how much should key token $j$ advance the contextual-position counter?

The reference implementation reuses the already scaled and masked attention logits and applies sigmoid:

$$
G=\sigma(L).
$$

Future positions have masked logits of negative infinity, so their gates become zero. Reusing $L$ avoids a second full query/key matrix multiplication.

#### Exactly where $[0,0,1,0,0]$ and $[1,1,1,0,0]$ come from

Suppose the five tokens are:

```text
Alice | slept | . | Bob | runs
```

After embeddings and the preceding transformer computation, they are represented by a hidden-state matrix:

$$
H\in\mathbb{R}^{5\times d_{\text{model}}}.
$$

There is one row $h_j$ per token. The rows are contextual vectors learned by the network; nobody manually labels a row as "person" or "sentence boundary."

For one attention head, learned projection matrices turn the hidden rows into queries and keys:

$$
Q=HW_Q,
\qquad
K=HW_K.
$$

Here:

- $W_Q\in\mathbb{R}^{d_{\text{model}}\times d_h}$ is a trained query projection;
- $W_K\in\mathbb{R}^{d_{\text{model}}\times d_h}$ is a trained key projection;
- $Q,K\in\mathbb{R}^{5\times d_h}$;
- $q_i$ is the query row for the token currently doing the looking;
- $k_j$ is the key row belonging to candidate token $j$.

In base CoPE, the ordinary attention keys can also be the **gate keys**. A variant gives gating its own trained projection:

$$
K^g=HW_g,
\qquad
W_g\in\mathbb{R}^{d_{\text{model}}\times d_h}.
$$

The weights $W_Q$, $W_K$, and, when present, $W_g$ are trained. A gate row is not a stored parameter: it is newly calculated from the current input's hidden states.

For the query at row 5, assume a two-dimensional toy head has produced:

$$
q_5=\begin{bmatrix}1&0\end{bmatrix},
$$

and these five gate-key rows:

$$
K^g=
\begin{bmatrix}
-5&0\\
-4&1\\
6&-1\\
-4&1\\
-5&0
\end{bmatrix}.
$$

The query is compared with every visible gate key:

$$
s_5=q_5(K^g)^{\mathsf T}
=\begin{bmatrix}-5&-4&6&-4&-5\end{bmatrix}.
$$

Applying sigmoid independently to every number gives:

$$
g_5=\sigma(s_5)
\approx
\begin{bmatrix}0.007&0.018&0.998&0.018&0.007\end{bmatrix}.
$$

Rounded only for a simple diagram, the **gate row** is:

$$
g_5\approx\begin{bmatrix}0&0&1&0&0\end{bmatrix}.
$$

The high gate at the period says that this token should advance this head's contextual-position counter. CoPE now reverse-cumulatively sums the gates. Using the rounded values:

$$
\begin{aligned}
p_{5,5}&=g_{5,5}=0,\\
p_{5,4}&=g_{5,4}+g_{5,5}=0,\\
p_{5,3}&=g_{5,3}+g_{5,4}+g_{5,5}=1,\\
p_{5,2}&=g_{5,2}+g_{5,3}+g_{5,4}+g_{5,5}=1,\\
p_{5,1}&=g_{5,1}+g_{5,2}+g_{5,3}+g_{5,4}+g_{5,5}=1.
\end{aligned}
$$

That produces the **contextual-position row**:

$$
p_{5,:}\approx\begin{bmatrix}1&1&1&0&0\end{bmatrix}.
$$

So the complete distinction is:

```text
learned query/key comparison → gate row       ≈ [0, 0, 1, 0, 0]
reverse cumulative sum       → position row   ≈ [1, 1, 1, 0, 0]
```

Neither row is a hand-written label or the output of softmax. The gate row is a simplified picture of five independent sigmoid outputs. In real execution the gates and contextual positions remain soft fractions; the model does not normally threshold them into exact zeros and ones.

The sigmoid/softmax distinction is important:

```text
gate scores      → sigmoid one by one → how much each token advances the counter
final logits     → softmax as a row   → weights summing to 1 for mixing V
```

Sigmoid permits many gates in a row to be high simultaneously. Softmax would make entries compete and sum to one, preventing the row from independently marking several events.

No trainer supplies either toy row as the target. Initially the projections can produce unhelpful gates. The next-token loss flows backward through the value mixture, attention softmax, CoPE position score, cumulative sum, sigmoid, dot products, and projection matrices. Training gradually makes gate patterns useful when they reduce prediction loss. A real head might learn to count sentence boundaries, words, code statements, or something with no simple human label; these binary-looking rows are only constructed examples.

#### Contextual-position matrix

For every query row, CoPE reverse-cumulatively sums its gates:

$$
p_{i,j}=\sum_{k=j}^{i}g_{i,k}.
$$

This forms:

$$
P^{\text{ctx}}\in\mathbb{R}^{T\times T}.
$$

Important special cases are:

- If every visible gate is 1, then $p_{i,j}=i-j+1$, which recovers ordinary token-counting relative position under the paper's indexing convention.
- If irrelevant tokens receive gates near 0, they contribute almost no contextual distance.
- If only sentence boundaries receive gates near 1, $p_{i,j}$ approximately counts sentences.
- Because sigmoid produces continuous values, contextual positions can be fractional.

The contextual position is query-dependent and head-dependent. The same key token can receive different contextual positions from two different queries or heads.

#### Maximum contextual position

Without a limit, contextual positions could range up to the token context length $T$, requiring $T+1$ position embeddings. CoPE caps them:

$$
p_{i,j}\leftarrow\min(p_{i,j},p_{\max}).
$$

All positions beyond $p_{\max}$ therefore share the final position representation. This keeps the learned table and position-logit computation small. It is a deliberate compression tradeoff: distances beyond the cap are treated as simply "far enough."

#### Learned position table and fractional interpolation

CoPE learns integer-position embeddings:

$$
E=
\begin{bmatrix}
e[0]&e[1]&\cdots&e[p_{\max}]
\end{bmatrix},
\qquad
E\in\mathbb{R}^{d_h\times(p_{\max}+1)}.
$$

For a fractional position $p$, let:

$$
f=\lfloor p\rfloor,
\qquad
c=\lceil p\rceil,
\qquad
w=p-f.
$$

The conceptual embedding interpolation is:

$$
e[p]=(1-w)e[f]+we[c].
$$

For example, if $p=1.6$:

$$
e[1.6]=0.4e[1]+0.6e[2].
$$

The conceptual final logit is:

$$
\ell_{i,j}
=
q_i^{\mathsf T}k_j
+
q_i^{\mathsf T}e[p_{i,j}],
$$

followed by softmax across the key index $j$.

#### Efficient interpolation used by the paper

Materializing a different $d_h$-dimensional vector $e[p_{i,j}]$ for every query/key pair would be expensive. The paper first computes query scores for all integer position embeddings:

$$
Z=QE,
\qquad
Z\in\mathbb{R}^{T\times(p_{\max}+1)}.
$$

Here:

$$
Z_{i,p}=q_i^{\mathsf T}e[p].
$$

For contextual position $p_{i,j}$, it interpolates the two scalar entries from row $i$ of $Z$:

$$
z_i[p_{i,j}]
=
(1-w)Z_{i,f}+wZ_{i,c}.
$$

The final attention logits are:

$$
L'_{i,j}=L_{i,j}+z_i[p_{i,j}].
$$

Then:

$$
A=\operatorname{softmax}(L'),
\qquad
O=AV.
$$

This produces the same interpolated positional score without constructing a position vector for every pair.

#### Multi-head and multi-layer behavior

Every attention head has different queries and keys, so every head produces its own gate and contextual-position matrices:

$$
G^{(h)},P_{\text{ctx}}^{(h)}\in\mathbb{R}^{T\times T}.
$$

One head can count tokens, another word beginnings, another sentence boundaries, and another code or document structure. No labels explicitly assign those roles; the task loss determines what is useful.

In the paper's base design, the integer position embeddings can be shared across heads. The authors also experimented with sharing them across layers. Sharing reduces parameters, while head-specific queries still create different positional measurements.

#### What is trained

The following are trainable:

- $W_Q$ and $W_K$, which determine the query/key features and therefore the gates;
- the integer contextual-position embedding table $E$;
- $W_V$, output projections, and the rest of the transformer.

The following are computed activations, not permanent parameter tables:

- gate matrix $G$;
- contextual-position matrix $P^{\text{ctx}}$;
- interpolated position logits for a specific input.

Sigmoid, cumulative sum, interpolation weights, softmax, and matrix multiplication form a differentiable path. Backpropagation from the language-model loss can therefore teach which tokens should open gates and which contextual positions a query should prefer. The model is not given sentence-boundary labels as part of CoPE itself.

#### Gate-design variants

The base method reuses the ordinary keys:

$$
g_{i,j}=\sigma(q_i^{\mathsf T}k_j).
$$

This is efficient, but it couples "this token should be attended to" with "this token should count as a positional step." The paper evaluates alternatives:

1. **Separate gate keys:**

   $$
   k_j^{g}=W_gh_j,
   \qquad
   g_{i,j}=\sigma(q_i^{\mathsf T}k_j^g).
   $$

   This lets counting and content retrieval use different key representations, but adds a projection and computation. Here $W_g$ and $W_K$ are normally different trained matrices. If they were identical, then $K^g=K$ and storing both would be redundant.

2. **Value gates:**

   $$
   g_{i,j}=\sigma(q_i^{\mathsf T}v_j).
   $$

   This also separates gating from ordinary keys but cannot fully reuse the original $QK^{\mathsf T}$ logits.

3. **CoPE with ordinary position methods:** CoPE can be combined with relative position or RoPE; its contextual score is an additional positional signal rather than a mandatory exclusive replacement.

4. **CoPE-ALiBi:** the contextual position can be multiplied by a head-specific slope instead of using the learned embedding table.

5. **CoPE-MLP:** an MLP can map contextual positions more flexibly, but the paper reports a much larger training memory/runtime cost for that experimental variant.

#### Does CoPE increase the KV cache?

It depends on which gate design is used:

1. **Base CoPE reusing ordinary keys:**

   $$
   K^g=K,
   \qquad
   g_{i,j}=\sigma(q_i^{\mathsf T}k_j).
   $$

   There is no separate $W_g$ and no separate gate-key cache. Each old $k_j$ is stored once and reused both for ordinary content attention and for the CoPE gate. The cache remains the ordinary $K$ and $V$ cache. CoPE still adds temporary gate/cumulative-sum computation, but not another persistent key vector per cached token.

2. **CoPE with separate gate keys:**

   $$
   K=HW_K,
   \qquad
   K^g=HW_g,
   \qquad
   W_g\ne W_K.
   $$

   Decode must compare each new query with the gate keys of previous tokens. A practical implementation therefore needs to cache $K^g$ as well as $K$ and $V$, unless it chooses the much more expensive option of retaining/recomputing old hidden states and projecting them again.

   For one head, if $K$, $V$, and $K^g$ all have dimension $d_h$ and use $s$ bytes per element:

   $$
   \text{ordinary KV bytes per token per layer}=2d_hs,
   $$

   while:

   $$
   \text{separate-gate bytes per token per layer}=3d_hs.
   $$

   Under those equal-dimension assumptions, the per-token cache is 50% larger than the ordinary KV cache. The exact increase depends on the number of ordinary KV heads, gate heads, their dimensions, quantization, and whether anything is shared:

   $$
   M_{\text{token,layer}}
   =
   \left(2N_{\text{KV}}d_h+N_gd_g\right)s.
   $$

   Here $N_{\text{KV}}$ is the number of ordinary KV heads, $N_g$ is the number of separately cached gate-key heads, and $d_g$ is their dimension.

3. **Value-gate variant:**

   If gates use the already cached values,

   $$
   g_{i,j}=\sigma(q_i^{\mathsf T}v_j),
   $$

   no separate gate-key cache is required, although extra score computation is still needed.

The projection matrix $W_g$ itself is a fixed model-weight tensor stored once per relevant layer. It does not grow with context length. The projected gate keys $k_j^g$ are per-token activations, so those are what make a separate-gate cache grow as more tokens are generated.

#### Prefill execution

During prefill for one layer and head:

```text
1. Compute Q, K, V for all prompt tokens.
2. Compute the causal T × T content-logit matrix L.
3. Apply sigmoid to L to obtain gates G.
4. Reverse-cumulatively sum every gate row to obtain Pctx.
5. Clamp Pctx to pmax.
6. Compute Z = QE for all queries and integer contextual positions.
7. Gather and interpolate Z using Pctx.
8. Add the contextual position logits to L.
9. Apply softmax and multiply by V.
```

This remains a full-attention operation over token pairs.

#### Cached decode execution

For a new token at decode time:

```text
1. Compute the new q, k, and v at the current layer.
2. Compare the new q with all cached keys to obtain one content-logit row.
3. Apply sigmoid to that row to obtain one new gate row.
4. Reverse-cumulatively sum and clamp that row to obtain contextual positions.
5. Compute the new query's scores against the integer position table E.
6. Gather/interpolate contextual position scores.
7. Add them to the content logits, apply softmax, and mix cached values.
8. Append the new key and value to the ordinary KV cache.
```

Old contextual-position rows cannot be treated as one universal per-token position because CoPE positions depend on the current query. However, old rows are not needed to generate the new output; only the new query's row is required.

#### Compute and memory complexity

For full attention with sequence length $T$ and head dimension $d_h$:

$$
QK^{\mathsf T}:\quad O(T^2d_h).
$$

CoPE can reuse these logits for its gates. Sigmoid and cumulative sums add approximately $O(T^2)$ elementwise work. The integer-position score matrix costs:

$$
QE:\quad O(Tp_{\max}d_h).
$$

Gathering/interpolating one positional score for every query/key pair adds $O(T^2)$ elementwise work.

Therefore CoPE does not change full attention from quadratic to linear complexity. It also does not reduce the ordinary KV-cache growth. Its learned position table is comparatively small when $p_{\max}$ is small, and the efficient formulation avoids storing a $d_h$-dimensional position vector for every token pair.

At decode time, CoPE still reads the cached keys and values for the new query. It adds one sigmoid/cumulative-sum row and the $p_{\max}$ position-score calculation; it is an accuracy/representation technique, not a KV-cache compression technique.

#### Empirical evidence and scope

The original paper reports strong results on selective-copy, Flip-Flop, and counting tasks, where ignoring irrelevant tokens or counting meaningful events is central. It also reports modest perplexity improvements on its Wikipedia language-modeling experiment and a small code model. CoPE was tested alone and in combinations with existing position methods.

Those results establish that the mechanism can learn useful contextual counting in the tested settings. They do not establish that CoPE is universally better for production-scale LLMs. The paper explicitly states that it did not test billion-parameter models, and its larger language/code experiments are much smaller than frontier-model pretraining.

#### Limitations and practical cautions

- CoPE still performs full token-pair attention and retains the normal KV-cache scaling.
- A finite $p_{\max}$ deliberately merges all sufficiently distant contextual positions.
- Reusing ordinary keys couples retrieval importance with positional counting; separate gates reduce that coupling but cost more.
- The mechanism learns statistical gates; it is not guaranteed to recover human-interpretable sentences, words, or program structures in every head.
- Query-dependent positions require extra kernel logic beyond standard RoPE or relative bias.
- It cannot normally be enabled on an already trained RoPE model solely through an inference flag.
- The original evidence is promising but limited in model scale, so serving-engine support and real-workload benchmarks must be checked.

#### What CoPE changes and does not change

| Component | Does CoPE change it? |
|---|---|
| Query/key/value projections | Uses them; may add a separate gate-key projection in a variant |
| Attention logits | Yes, adds contextual-position scores |
| Causal mask | No |
| Softmax and value mixture | No |
| KV-cache asymptotic size | No |
| Full-attention asymptotic complexity | No |
| Definition of positional distance | Yes, from token count to learned gate accumulation |
| Need for training | Yes, it must be part of the trained architecture |

## The useful evolutionary map

```text
One fixed encoder vector
        ↓
Additive attention: decoder searches encoder states
        ↓
Dot-product attention: compare query and key by multiplication
        ↓
Transformer self-attention: tokens search tokens in the same sequence
        ↓
Multi-head attention: several learned searches at once
        ↓
Three efficiency branches
        ├── Reduce the KV cache: MQA → GQA → MLA
        ├── Restrict retrieval: local/sliding-window/sparse attention
        └── Compress history into state:
            Linear Attention → DeltaNet → Gated DeltaNet → KDA
```

This is not one perfectly straight historical chain. The branches solve different problems.

## Full softmax attention and its KV-efficient variants

In **Multi-Head Attention (MHA)**, each query head has its own key and value head:

$$
N_{\text{Q-heads}}=N_{\text{KV-heads}}
$$

In **Multi-Query Attention (MQA)**, every query head shares one key/value head:

$$
N_{\text{KV-heads}}=1
$$

In **Grouped-Query Attention (GQA)**, groups of query heads share key/value heads:

$$
1<N_{\text{KV-heads}}<N_{\text{Q-heads}}
$$

The progression is therefore:

```text
MHA → highest KV-cache size; separate K and V for every query head
GQA → fewer shared KV heads; quality/efficiency compromise
MQA → one shared KV head; smallest cache of these three
```

**Multi-Head Latent Attention (MLA)** compresses the key/value representation into a smaller latent vector:

$$
c_t^{KV}=h_tW_{\text{down}}
$$

Conceptually, keys and values can be obtained from this latent representation:

$$
k_t=c_t^{KV}W_{\text{up},K},
\qquad
v_t=c_t^{KV}W_{\text{up},V}
$$

MLA still performs global softmax retrieval over previous tokens. Its cache still grows with sequence length, but the stored representation per token is much smaller.

## KV-cache memory

The approximate key/value memory added by one token is:

$$
M_{\text{KV/token}}
=
2N_{\text{layers}}N_{\text{KV-heads}}d_{\text{head}}s_{\text{KV}}
$$

Here:

- The factor \(2\) represents one key and one value.
- \(N_{\text{layers}}\): number of transformer layers containing attention.
- \(N_{\text{KV-heads}}\): number of key/value heads in each layer. This is found in the model configuration; it is not necessarily the number of query heads.
- \(d_{\text{head}}\): features in each head.
- \(s_{\text{KV}}\): bytes used to store one key/value element.

For requests with sequence lengths \(S_i\), total KV-cache memory is approximately:

$$
M_{\text{KV,total}}
=
M_{\text{KV/token}}
\sum_i S_i
$$

The model weights are shared across all requests, but every request has its own KV cache. For the full worked example and how this is built from the model config, see [04-memory-planning](./04-memory-planning.md).

## FlashAttention is an implementation, not a new attention rule

FlashAttention calculates the same exact softmax-attention result using tiles. It avoids materializing the complete token-by-token attention matrix in high-bandwidth memory:

```text
Load small Q, K, and V tiles into fast on-chip memory
                        ↓
Calculate partial scores and the running softmax
                        ↓
Immediately combine with V and write the final output
```

Therefore:

```text
MHA/GQA/MLA/KDA → change the model architecture
FlashAttention  → changes how an attention calculation is executed
```

## Linear attention and fixed-size state

Softmax attention explicitly compares queries with previous keys. Linear attention uses a feature function \(\phi\) that allows the multiplication to be reassociated:

$$
S_t=S_{t-1}+\phi(k_t)v_t^{\mathsf T}
$$

$$
z_t=z_{t-1}+\phi(k_t)
$$

$$
o_t
=
\frac{\phi(q_t)^{\mathsf T}S_t}
{\phi(q_t)^{\mathsf T}z_t}
$$

Instead of storing a separate key and value for every previous token, it keeps the fixed-size state \(S_t\). This is efficient, but several associations can interfere because they share the same limited state.

## DeltaNet, Gated DeltaNet, and KDA

DeltaNet first asks what the current state already returns for a key:

$$
\widehat v_t=S_{t-1}^{\mathsf T}k_t
$$

It calculates the correction:

$$
e_t=v_t-\widehat v_t
$$

It then updates the association:

$$
S_t=S_{t-1}+\beta_tk_te_t^{\mathsf T}
$$

Here, \(\beta_t\) controls the strength of the write. Unlike purely additive linear attention, DeltaNet writes the difference between the desired value and what is already stored.

Gated DeltaNet adds a learned forgetting value \(\alpha_t\):

$$
\widetilde S_t=\alpha_tS_{t-1}
$$

It then applies the DeltaNet correction to \(\widetilde S_t\). This allows old memory to decay as well as allowing a specific association to be corrected.

**Kimi Delta Attention (KDA)** replaces the coarse scalar decay with channel-specific decay:

$$
S_t
=
\left(I-\beta_tk_tk_t^{\mathsf T}\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_tk_tv_t^{\mathsf T}
$$

$$
o_t=S_t^{\mathsf T}q_t
$$

Different state features can therefore forget at different rates. KDA does not preserve every previous key/value vector separately; it preserves their learned effect in a fixed-size state.

## Why hybrid attention is useful

```text
Full softmax/MLA
    + direct retrieval from any retained token
    - cache and attention work grow with context length

Linear attention/KDA
    + fixed-size recurrent state
    + efficient long-context decoding
    - history is compressed and exact facts can interfere
```

A hybrid model uses mostly KDA layers and periodic MLA layers. KDA carries efficient recurrent memory, while MLA provides occasional direct global retrieval. Kimi Linear uses a 3:1 KDA-to-MLA pattern.

## Choosing attention for a short-context SQL generator

A short context does not create a stronger reason to use KDA. It makes ordinary softmax attention cheaper because the attention matrix and KV cache remain small. KDA's principal efficiency advantage appears at long context lengths, where it replaces a linearly growing KV cache with a fixed-size recurrent state.

For a specialized SQL generator, exact retrieval can be particularly valuable: the model may need to preserve precise table names, column names, joins, filters, and schema relationships. Compressing all of that information into a fixed-size KDA state can introduce interference. For reliably short prompts, standard attention with GQA is generally the simpler starting point.

```text
Reliably short context + exact schema retrieval
    → standard softmax attention or GQA is usually sufficient

Very long context or many long concurrent requests
    → KDA or a KDA/MLA hybrid becomes more attractive
```

KDA is a model architecture, not an inference setting that can normally be enabled on an existing softmax-attention model. Replacing attention layers with KDA changes the model computation and usually requires architecture-aware training or substantial continued training. Reducing the configured context length alone does not convert a model to KDA.

## Complexity summary

Let \(T\) be the current context length and \(w\) a local-attention window size.

| Mechanism | Decode memory as \(T\) grows | Attention work for one new token | Main purpose |
|---|---:|---:|---|
| MHA | \(O(T)\), largest KV constant | \(O(T)\) | Full direct retrieval |
| GQA/MQA | \(O(T)\), smaller KV constant | \(O(T)\) | Reduce cache traffic |
| MLA | \(O(T)\), compressed per-token cache | \(O(T)\) | Preserve global retrieval with a smaller cache |
| Sliding-window attention | \(O(w)\) | \(O(w)\) | Bound work to recent context |
| Linear attention/KDA | Fixed-size recurrent state | Constant with respect to \(T\) | Efficient long-context memory |

The state operation in KDA still has a real cost determined by its state dimensions. "Constant" here means that it does not increase merely because the context contains more tokens.

---

**Continue to:** [11-training-adaptation](./11-training-adaptation.md) for LoRA, QLoRA, distillation, and gradient averaging.
