"""
utils.py — Small helper utilities.
"""

from __future__ import annotations

import os

import numpy as np

import config
from data import Vocabulary


def save_embeddings(
    model,
    vocab: Vocabulary,
    out_dir: str = config.EMBED_DIR,
) -> None:
    """Save trained embeddings to ``out_dir`` in two formats.

    1. NumPy binary (``embeddings.npy`` + ``vocab.txt``) —
       fast to reload for downstream use.
    2. Plain-text (``embeddings.txt``) — one line per word,
       compatible with tools that read GloVe-format files.

    Parameters
    ----------
    model : SkipGramNS
        Trained model.
    vocab : Vocabulary
    out_dir : str
        Output directory (created if necessary).
    """
    os.makedirs(out_dir, exist_ok=True)

    embeddings = model.get_embeddings()

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for idx in range(len(vocab)):
            f.write(vocab.id2word[idx] + "\n")

    txt_path = os.path.join(out_dir, "embeddings.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx in range(len(vocab)):
            word = vocab.id2word[idx]
            vec_str = " ".join(f"{v:.6f}" for v in embeddings[idx])
            f.write(f"{word} {vec_str}\n")

    print(f"[utils] Embeddings saved to {out_dir}/")


def load_embeddings(
    embed_dir: str = config.EMBED_DIR,
) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    """Load previously saved embeddings.

    Parameters
    ----------
    embed_dir : str
        Directory containing ``embeddings.npy`` and ``vocab.txt``.

    Returns
    -------
    embeddings : np.ndarray, shape ``(V, D)``
    word2id : dict[str, int]
    id2word : dict[int, str]
    """
    embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))

    word2id: dict[str, int] = {}
    id2word: dict[int, str] = {}
    vocab_path = os.path.join(embed_dir, "vocab.txt")
    with open(vocab_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.strip()
            word2id[word] = idx
            id2word[idx] = word

    return embeddings, word2id, id2word
