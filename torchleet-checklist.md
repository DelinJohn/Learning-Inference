# TorchLeet checklist

Problems are from [TorchLeet](https://github.com/Exorust/TorchLeet) by Chandrahas Aroori ([@Exorust](https://github.com/Exorust)) and contributors, MIT licensed. Repo cloned at `Torchleet/`. Tick a box when your notebook passes `check()`.

**Rules for yourself**
1. No AI and no solutions. Stuck? Re-read the problem, print tensor shapes, and read the PyTorch docs. Solutions are hidden in `~/.torchleet-solutions/`.
2. After each problem, add 3–5 lines to your notes on what you got wrong.
3. Re-do attention, GQA and KV cache from a blank file a week later.

**Setup:** use the existing `timeseriesllm` environment (torch 2.5.1 + CUDA, Triton). Run `jupyter lab` from the `Torchleet/` folder and pick the `timeseriesllm` kernel.

**Time:** the repo's own budgets (basic 20m, easy 30m, medium 45m, hard 90m, expert 2h) put the LLM path (now 23 problems) at about 23 h. That is lower than my earlier 35–45 h estimate. Time your first 3 problems and use your own average.

---
## Phase 1 — LLM path: foundations and core transformer (do first)
9 problems · ~7 h (repo budgets)

- [ ] **Implement Byte Pair Encoding from Scratch** — easy, ~30m, graded  
  [question](Torchleet/llm/Byte-Pair-Encoder/BPE-q3.ipynb)  
  Ties to your notes: tokens in/out of the model
- [ ] **Implement Sinusoidal Embeddings** — medium, ~45m, graded  
  [question](Torchleet/llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb)  
  Ties to your notes: positional encoding (original Transformer)
- [ ] **Implement ROPE Embeddings** — medium, ~45m, graded  
  [question](Torchleet/llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb)  
  Ties to your notes: RoPE - rotates Q/K; compare with ALiBi
- [ ] **Implement Attention from Scratch** — medium, ~45m, graded  
  [question](Torchleet/llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb)  
  Ties to your notes: Q/K/V, softmax(QK^T/sqrt(d))V, causal mask
- [ ] **Implement Multi-Head Attention from Scratch** — medium, ~45m, graded  
  [question](Torchleet/llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb)  
  Ties to your notes: heads, [B,T,H,d_head], W_O
- [ ] **Implement Grouped Query Attention from Scratch** — medium, ~45m, graded  
  [question](Torchleet/llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb)  
  Ties to your notes: H_q vs H_kv, MHA/GQA/MQA table
- [ ] **Implement KV Cache for Autoregressive Generation** — medium, ~45m, graded  
  [question](Torchleet/v3/llm-inference/kv-cache/kv-cache.ipynb)  
  Ties to your notes: M_KV/token = 2*N_layers*N_kv*d_head*bytes; decode reads cache
- [ ] **Implement Sliding Window Attention** — medium, ~45m, graded  
  [question](Torchleet/v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb)  
  Ties to your notes: O(w) memory, local attention branch
- [ ] **Implement SmolLM from Scratch** — hard, ~90m, graded  
  [question](Torchleet/llm/SmolLM/smollm-q12-Question.ipynb)  
  Ties to your notes: whole decoder: prefill + decode in one model

## Phase 2 — LLM path: full model, LoRA, decoding, inference (closest to your notes)
9 problems · ~10 h (repo budgets)

- [ ] **Implement LoRA on a Linear Layer** — medium, ~45m, graded  
  [question](Torchleet/v3/alignment-training/lora/lora.ipynb)  
  Ties to your notes: y = xW + xBA, r(d_in+d_out) params
- [ ] **Implement Temperature Sampling** — easy, ~30m, graded  
  [question](Torchleet/v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb)  
  Ties to your notes: sampling after the LM head logits
- [ ] **Implement Top-k Sampling** — medium, ~45m, graded  
  [question](Torchleet/v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb)  
  Ties to your notes: sampling
- [ ] **Implement Top-p (Nucleus) Sampling** — medium, ~45m, graded  
  [question](Torchleet/v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb)  
  Ties to your notes: sampling
- [ ] **Implement Speculative Decoding** — hard, ~90m, graded  
  [question](Torchleet/v3/llm-inference/speculative-decoding/speculative-decoding.ipynb)  
  Ties to your notes: multi-token decoding; attacks memory-bound decode
- [ ] **Implement Continuous Batching for LLM Inference** — hard, ~90m, graded  
  [question](Torchleet/v3/llm-inference/continuous-batching/continuous-batching.ipynb)  
  Ties to your notes: iteration-level batching, vLLM scheduler
- [ ] **Build a Complete LLM Inference Engine** — expert, ~120m, graded  
  [question](Torchleet/v3/llm-inference/inference-engine/inference-engine.ipynb)  
  Ties to your notes: prefill/decode, KV blocks, batching all together
- [ ] **Implement Mixture of Experts Layer** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb)  
  Ties to your notes: P_active vs P total; Kimi K3 router (898 experts)
- [ ] **Implement RMS Norm** — easy, ~30m, not graded  
  [question](Torchleet/llm/RMS-Norm/rms-norm.ipynb)  
  Ties to your notes: normalisation used in modern LLMs

## Phase 3 — LLM path: alignment (optional until you need it)
3 problems · ~5 h (repo budgets)

- [ ] **Implement DPO Loss from Scratch** — hard, ~90m, graded  
  [question](Torchleet/v3/alignment-training/dpo/dpo.ipynb)  
  Ties to your notes: alignment (not in notes yet)
- [ ] **Implement PPO for RLHF** — hard, ~90m, graded  
  [question](Torchleet/v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb)  
  Ties to your notes: alignment (not in notes yet)
- [ ] **Implement GRPO (DeepSeek-R1 Algorithm)** — expert, ~120m, graded  
  [question](Torchleet/v3/alignment-training/grpo/grpo.ipynb)  
  Ties to your notes: alignment (not in notes yet)

## Phase 4 — Advanced, inference and systems relevant
8 problems · ~12 h (repo budgets)

- [ ] **Implement Beam Search for LLM Decoding** — medium, ~45m, graded  
  [question](Torchleet/v3/llm-decoding/beam-search/beam-search.ipynb)  
  Ties to your notes: decoding
- [ ] **Implement Selective State Space Model (Mamba)** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/mamba/mamba.ipynb)  
  Ties to your notes: state-space / fixed-size state, like KDA branch
- [ ] **Write a Fused Softmax Kernel in Triton** — expert, ~120m, graded  
  [question](Torchleet/v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb)  
  Ties to your notes: kernel fusion, memory-bound ops
- [ ] **Implement FlashAttention-2 in Triton** — expert, ~120m, graded  
  [question](Torchleet/v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb)  
  Ties to your notes: tiled attention, avoids N x N matrix in HBM
- [ ] **Implement Ring Attention for Long Contexts** — expert, ~120m, graded  
  [question](Torchleet/v3/gpu-systems/ring-attention/ring-attention.ipynb)  
  Ties to your notes: long context across GPUs
- [ ] **Implement FSDP from Scratch** — expert, ~120m, graded  
  [question](Torchleet/v3/gpu-systems/fsdp/fsdp.ipynb)  
  Ties to your notes: sharding weights across GPUs
- [ ] **Implement Gradient Checkpointing** — hard, ~90m, graded  
  [question](Torchleet/v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb)  
  Ties to your notes: trade compute for memory
- [ ] **Implement Knowledge Distillation** — medium, ~45m, graded  
  [question](Torchleet/v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb)  
  Ties to your notes: teacher -> student, KL loss

## Phase 5 — PyTorch basics (skip anything you already know, or do first if rusty)
21 problems · ~10 h (repo budgets)

- [ ] **Implement Linear Regression** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/lin-regression/lin-regression.ipynb)
- [ ] **Write a Custom Dataset and DataLoader** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/custom-dataset/custom-dataset.ipynb)
- [ ] **Write a Custom Activation Function** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/custom-activation/custom-activation.ipynb)
- [ ] **Implement Custom Loss Function (Huber Loss)** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/custom-loss/custom-loss.ipynb)
- [ ] **Implement a Deep Neural Network** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/custom-DNN/custon-DNN.ipynb)
- [ ] **Visualize Training with TensorBoard** — basic, ~20m, graded  
  [question](Torchleet/torch/basic/tensorboard/tensorboard.ipynb)
- [ ] **Save and Load PyTorch Model** — basic, ~20m, not graded  
  [question](Torchleet/torch/basic/save-model/save_model.ipynb)
- [ ] **Implement a CNN on CIFAR-10** — easy, ~30m, graded  
  [question](Torchleet/torch/easy/cnn/CNN.ipynb)
- [ ] **Implement an RNN from Scratch** — easy, ~30m, graded  
  [question](Torchleet/torch/easy/rnn/RNN.ipynb)
- [ ] **Data Augmentation with torchvision.transforms** — easy, ~30m, not graded  
  [question](Torchleet/torch/easy/augmentation/augmentation.ipynb)
- [ ] **Add Benchmarking to PyTorch Code** — easy, ~30m, graded  
  [question](Torchleet/torch/easy/benchmark/bench.ipynb)
- [ ] **Train an Autoencoder for Anomaly Detection** — easy, ~30m, graded  
  [question](Torchleet/torch/easy/autoencoder/autoencoder.ipynb)
- [ ] **Quantize Your Language Model** — easy, ~30m, graded  
  [question](Torchleet/torch/easy/quantize-lm/quantize-language-model.ipynb)
- [ ] **Mixed Precision Training with torch.cuda.amp** — easy, ~30m, not graded  
  [question](Torchleet/torch/easy/cuda-amp/cuda-amp.ipynb)
- [ ] **CNN Parameter Initialization** — medium, ~45m, graded  
  [question](Torchleet/torch/medium/cnn-param-init/CNN_ParamInit.ipynb)
- [ ] **Implement a CNN from Scratch** — medium, ~45m, graded  
  [question](Torchleet/torch/medium/cnn-scratch/CNN_scratch.ipynb)
- [ ] **Implement an LSTM from Scratch** — medium, ~45m, graded  
  [question](Torchleet/torch/medium/lstm/LSTM.ipynb)
- [ ] **Implement Softmax from Scratch** — easy, ~30m, graded  
  [question](Torchleet/v3/classical-ml/softmax/softmax.ipynb)
- [ ] **Implement K-Means Clustering in PyTorch** — easy, ~30m, graded  
  [question](Torchleet/v3/classical-ml/kmeans/kmeans.ipynb)
- [ ] **Implement KNN in PyTorch** — easy, ~30m, graded  
  [question](Torchleet/v3/classical-ml/knn/knn.ipynb)
- [ ] **Implement Logistic Regression with Gradient Descent** — easy, ~30m, graded  
  [question](Torchleet/v3/classical-ml/logistic-regression/logistic-regression.ipynb)

## Phase 6 — Everything else in Advanced (optional)
16 problems · ~21 h (repo budgets)

- [ ] **Implement AlexNet from Scratch** — medium, ~45m, graded  
  [question](Torchleet/torch/medium/alexnet/alexnet.ipynb)
- [ ] **Train a 3D CNN for CT Image Segmentation** — medium, ~45m, graded  
  [question](Torchleet/torch/medium/3dcnn/3DCNN.ipynb)
- [ ] **Write a Custom Autograd Function (SILU)** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/custom-autograd/custom-autgrad-function.ipynb)
- [ ] **Write a Transformer from Scratch** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/transformer/transformer.ipynb)
- [ ] **Write a GAN** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/GAN/GAN.ipynb)
- [ ] **Sequence-to-Sequence with Attention** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/seq-seq/seq-to-seq-with-Attention.ipynb)
- [ ] **Explainable AI (GradCAM/SHAP)** — hard, ~90m, not graded  
  [question](Torchleet/torch/hard/xai/xai.ipynb)
- [ ] **Implement Adam from Scratch** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/optimizers/adam/adam.ipynb)
- [ ] **Implement AdamW from Scratch** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/optimizers/adamw/adamw.ipynb)
- [ ] **Implement Muon from Scratch** — hard, ~90m, graded  
  [question](Torchleet/torch/hard/optimizers/muon/muon.ipynb)
- [ ] **Implement Contrastive Loss (InfoNCE) + CLIP Training Loop** — medium, ~45m, graded  
  [question](Torchleet/v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb)
- [ ] **Implement 2D Positional Embeddings** — medium, ~45m, graded  
  [question](Torchleet/v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb)
- [ ] **Implement DDPM from Scratch** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/ddpm/ddpm.ipynb)
- [ ] **Implement DDIM Sampling + Classifier-Free Guidance** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb)
- [ ] **Implement Vision Transformer + MAE Pretraining** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/vit-mae/vit-mae.ipynb)
- [ ] **Implement a Swin Transformer Block (Shifted Window Attention)** — hard, ~90m, graded  
  [question](Torchleet/v3/modern-architectures/swin-transformer/swin-transformer.ipynb)

## Phase 7 — LLM path extras (were unlisted in the repo)
2 problems · ~1.5 h. Flash Attention is worth doing right after Phase 1: it is the tiled-attention idea from your notes.

- [ ] **Flash Attention** — medium, ~45m, graded  
  [question](Torchleet/llm/Flash-Attention/flash-attention.ipynb)  
  Ties to your notes: tiled attention, avoids the N x N matrix in HBM
- [ ] **Create Embeddings Out Of An LLM** — medium, ~45m, not graded  
  [question](Torchleet/llm/Create-Embeddings-out-of-an-LLM/embeddings-q2.ipynb)
