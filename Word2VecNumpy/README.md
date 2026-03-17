# Word2Vec from Scratch — Skip-Gram with Negative Sampling

A complete implementation of **Word2Vec (Skip-Gram with Negative Sampling)** in pure NumPy.  

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Mathematical Background](#mathematical-background)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Configuration](#configuration)  
7. [Evaluation](#evaluation)  
8. [References](#references)

---

## Overview

Word2Vec learns dense vector representations (embeddings) of words such that words appearing in similar contexts are mapped to nearby points in the embedding space.

This repository implements the **Skip-Gram** variant with **Negative Sampling (SGNS)**, as introduced by Mikolov et al. (2013). Given a center (target) word, the model predicts its surrounding context words while contrasting them against randomly drawn "negative" words.

### Key Features

| Feature | Details |
|---|---|
| Architecture | Skip-Gram |
| Objective | Negative Sampling (NCE-style) |
| Optimizer | SGD with linear learning-rate decay |
| Subsampling | Frequent-word subsampling (Mikolov et al., 2013) |
| Negative Distribution | Unigram distribution raised to the 3/4 power |
| Corpus | [WikiText-2](https://huggingface.co/datasets/wikitext) (downloaded automatically) |
| Evaluation | Cosine-similarity nearest neighbours, word analogy task |

---

## Repository Structure

```
word2vec/
├── README.md            # This file
├── requirements.txt     # Python dependencies (numpy only)
├── config.py            # All hyper-parameters in one place
├── data.py              # Corpus loading, tokenisation, vocabulary,
│                        #   subsampling, negative-sampling table,
│                        #   training-pair generation
├── model.py             # SkipGramNS model: embeddings, forward pass,
│                        #   loss, analytic gradients, parameter update
├── train.py             # Training loop with progress reporting
├── evaluate.py          # Nearest-neighbour search & analogy evaluation
├── utils.py             # Small helper utilities
└── main.py              # CLI entry-point: ties everything together
```

---

## Mathematical Background

### Objective

For a center word $w$ and a true context word $c^{+}$, with $K$ negative samples
$\{c_1^{-}, \dots, c_K^{-}\}$, the per-example SGNS loss is:

$$
\mathcal{L} = -\log\sigma(\mathbf{u}_{c^{+}}^{\top}\mathbf{v}_{w})
             - \sum_{k=1}^{K}\log\sigma(-\mathbf{u}_{c_k^{-}}^{\top}\mathbf{v}_{w})
$$

where $\mathbf{v}_w$ is the center embedding, $\mathbf{u}_c$ is the context embedding,
and $\sigma$ is the sigmoid function.

### Gradients

$$\frac{\partial\mathcal{L}}{\partial\mathbf{v}_w} = (\sigma(\mathbf{u}_{c^{+}}^{\top}\mathbf{v}_w)-1)\,\mathbf{u}_{c^{+}} + \sum_{k=1}^{K}\sigma(\mathbf{u}_{c_k^{-}}^{\top}\mathbf{v}_w)\,\mathbf{u}_{c_k^{-}}$$


$$
\frac{\partial\mathcal{L}}{\partial\mathbf{u}_{c^{+}}}
  = (\sigma(\mathbf{u}_{c^{+}}^{\top}\mathbf{v}_w)-1)\,\mathbf{v}_w
$$

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{u}_{c_k^{-}}} = \sigma(\mathbf{u}_{c_k^{-}}^{\top}\mathbf{v}_w)\,\mathbf{v}_w
$$

These are applied as vanilla SGD updates:

$$\theta \leftarrow \theta - \eta\,\nabla_{\theta}\mathcal{L}$$

---

## Installation

```bash
cd word2vec

# Create a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train from scratch

```bash
python main.py
```

This will:
1. Download and preprocess **WikiText-2** ($\approx 2$ M tokens).
2. Build a vocabulary (top `MAX_VOCAB_SIZE` words by frequency).
3. Train Skip-Gram with Negative Sampling for the configured number of epochs.
4. Save the learned embeddings to `embeddings/`.
5. Run nearest-neighbour and analogy evaluation.

### Custom options

All hyper-parameters are set in `config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `EMBED_DIM` | 100 | Dimensionality of word vectors |
| `WINDOW_SIZE` | 5 | Max context window (actual sampled uniformly in [1, W]) |
| `NUM_NEGATIVES` | 5 | Negative samples per positive pair |
| `EPOCHS` | 5 | Training epochs |
| `LEARNING_RATE` | 0.025 | Initial learning rate |
| `MIN_LR` | 1e-4 | Minimum learning rate (linear decay) |
| `MIN_COUNT` | 5 | Discard words with frequency < this |
| `SUBSAMPLE_THRESH` | 1e-3 | Subsampling threshold for frequent words |
| `BATCH_SIZE` | 256 | Mini-batch size for SGD |

---

## Evaluation

After training, the script automatically runs:

1. **Nearest neighbours** — prints the 10 closest words (by cosine similarity) for a curated list of query words.
2. **Word analogies** — tests `A : B :: C : ?` relationships (e.g. *king − man + woman = queen*).

---

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
3. Goldberg, Y., & Levy, O. (2014). *word2vec Explained*. arXiv:1402.3722.
4. Rong, X. (2014). *word2vec Parameter Learning Explained*. arXiv:1411.2738.
