# Build an LLM From Scratch 🧠🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

A comprehensive, step-by-step journey into the heart of Large Language Models. This repository contains the complete implementation of a GPT-style transformer model, moving from raw text processing to instruction fine-tuning and parameter-efficient adaptation (LoRA).

---

## 📖 Project Narrative

In an era of multi-billion parameter models, understanding the "magic" under the hood is more critical than ever. This project demystifies the transformer architecture by building every component from the ground up using Python and PyTorch.

We don't just use libraries; we implement the **Causal Self-Attention**, the **Layer Normalization**, the **Feed-Forward Networks**, and the **Training Pipelines** ourselves. By the end of this journey, you will have a deep, first-principles understanding of how GPT models like GPT-2 and GPT-3 actually work.

---

## 🏗️ Technical Architecture Highlights

### 1. Causal Self-Attention

The core of the transformer. We implement a multi-head attention mechanism that allows the model to weigh the importance of different tokens in a sequence. Crucially, we apply **Causal Masking** to ensure the model only looks at previous tokens during training—a defining characteristic of generative (decoder-only) LLMs.

### 2. Rotary Positional Embeddings (RoPE) 🔄

Going beyond the standard absolute positional embeddings found in early GPT models, this repository implements **RoPE**.

- **The Intuition**: RoPE encodes relative positions using a rotation matrix, allowing the model to better generalize to sequence lengths it hasn't seen during training.
- **Implementation**: See `rotaryEmbeddings.py` and `mulithead_attention_with_rotary_embeddings.py`.

### 3. Low-Rank Adaptation (LoRA) 📉

Fine-tuning a massive LLM is computationally expensive. We implement **LoRA**, a breakthrough technique that:

- Freezes the original weights.
- Injects small, trainable "adapter" matrices ($A$ and $B$) with a low rank ($r$).
- Significantly reduces the number of trainable parameters (often by >90%) while maintaining high performance.
- **Deep Dive**: Check `lowrankmatrixadaption.ipynb`.

### 4. Training Stability & Optimization

Modern LLM training requires more than just a simple SGD. This repo demonstrates:

- **Learning Rate Warmup**: Gradually increasing the learning rate to prevent early divergence.
- **Cosine Decay**: Smoothly decreasing the learning rate towards the end of training.
- **Gradient Clipping**: Scaling gradients to prevent "exploding gradient" issues.

---

## 🗺️ The Learning Path (Story of the Notebooks)

The project is structured as a progressive curriculum:

1.  **[1-text-processing.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/1-text-processing.ipynb)**: From raw strings to tokens. Implements simple tokenization, Byte Pair Encoding (BPE), and the `GPTDataset` for sliding-window data loading.
2.  **[2-coding-attention.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/2-coding-attention.ipynb)**: Step-by-step assembly of the Self-Attention mechanism, Causal Masking, and Multi-Head scaling.
3.  **[3-llm-architecture.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/3-llm-architecture.ipynb)**: Building the `GPTModel`. Implements LayerNorm, GELU activation, and the residual connections that make deep transformers possible.
4.  **[4-pretraining.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/4-pretraining.ipynb)**: Calculating loss, evaluating models, and the first pretraining loop on "The Verdict" (short story).
5.  **[5-finetuning.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/5-finetuning.ipynb)**: Adapting the pretrained model for Spam Classification. Demonstrates the power of transfer learning.
6.  **[6-instruction-finetuning.ipynb](file:///home/amh1k/ml and python/llm-from-scratch/6-instruction-finetuning.ipynb)**: The final frontier. Teaching the model to follow user commands using the Alpaca dataset format.

---

## 🔬 Repository Anatomy

### Core Model Logic

| File                                            | Purpose                                                      |
| :---------------------------------------------- | :----------------------------------------------------------- |
| `gpt_model.py`                                  | The main `GPTModel` class definition.                        |
| `multihead_attention.py`                        | Standard Multi-Head Attention module.                        |
| `rotaryEmbeddings.py`                           | Mathematical implementation of Rotary Positional Embeddings. |
| `mulithead_attention_with_rotary_embeddings.py` | Advanced attention using RoPE.                               |

### Data & Training

| File                           | Purpose                                                                            |
| :----------------------------- | :--------------------------------------------------------------------------------- |
| `dataloader_v1.py`             | PyTorch `Dataset` and `DataLoader` factory for text.                               |
| `model_training_utlilities.py` | Loss calculation, evaluation, and the main training loop (`train_model_simple`).   |
| `plot_losses.py`               | Visualization tool for training and validation loss curves.                        |
| `gpt_download.py`              | Script to fetch pretrained OpenAI GPT-2 weights.                                   |
| `load_weights_into_gpt.py`     | Logic to map OpenAI's TensorFlow checkpoints into our custom PyTorch architecture. |

### Utilities & Artifacts

- **Text Generation**: `generate_text.py` handles the decoding strategies (Temperature, Top-K).
- **Data Files**: `the-verdict.txt` (pretraining), `instruction-data.json` (SFT).
- **Models**: `.pth` files contain saved weights for the Spam Classifier and SFT model.

---

## 🚀 Getting Started

### 1. Requirements

```bash
pip install torch tiktoken matplotlib pandas
```

### 2. Usage: Load and Generate

You can load the GPT-2 weights into the custom model and start generating text:

```python
import torch
from gpt_model import GPTModel
from load_weights_into_gpt import load_weights_into_gpt
from gpt_download import download_and_load_gpt2

# 1. Download weights
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# 2. Initialize Model
model = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(model, params)

# 3. Generate
# (See generate_text.py for details)
```

---

## 🤝 Acknowledgments

Inspired by the "LLMs from Scratch" book and the broader open-source AI community. Special thanks to OpenAI for making GPT-2 weights public!
