#!/usr/bin/env python3
"""
main.py - Entry point for training Word2Vec (Skip-Gram + Negative Sampling).

Running this script will:

1. Download and tokenise the WikiText-2 corpus.
2. Build a vocabulary from the most frequent words.
3. Apply subsampling to reduce the weight of very common words.
4. Construct a unigram table for efficient negative sampling.
5. Initialise and train a Skip-Gram model with negative sampling.
6. Save the learned embeddings to disk.
7. Run nearest-neighbour and word-analogy evaluation.

All hyper-parameters are read from ``config.py``.
"""

from __future__ import annotations

import numpy as np

import config
from data import (
    build_neg_table,
    build_vocab,
    load_corpus,
    subsample,
)
from evaluate import nearest_neighbours, word_analogies
from evaluate import DEFAULT_ANALOGIES, DEFAULT_QUERIES
from model import SkipGramNS
from train import train


def main() -> None:
    """Full Word2Vec pipeline: load → build vocab → train → evaluate."""

    np.random.seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    # Load corpus
    print("=" * 60)
    print(" STEP 1 / 5 - Loading corpus")
    print("=" * 60)
    tokens = load_corpus()

    # Build vocabulary 
    print("\n" + "=" * 60)
    print(" STEP 2 / 5 - Building vocabulary")
    print("=" * 60)
    vocab = build_vocab(tokens)

    eval_words = set(DEFAULT_QUERIES)
    for a, b, c, d in DEFAULT_ANALOGIES:
        eval_words.update((a, b, c, d))
    missing_eval_words = sorted(w for w in eval_words if w not in vocab)
    if missing_eval_words:
        print(
            f"[main] Warning: {len(missing_eval_words)} eval words are OOV "
            f"after MIN_COUNT filtering: {missing_eval_words}"
        )

    # Convert corpus to integer IDs, discarding OOV words
    corpus_ids = np.array(
        [vocab.word2id[t] for t in tokens if t in vocab.word2id],
        dtype=np.int64,
    )
    print(f"[main] Corpus encoded: {len(corpus_ids):,} in-vocabulary tokens")

    # Subsample frequent words
    print("\n" + "=" * 60)
    print(" STEP 3 / 5 - Subsampling frequent words")
    print("=" * 60)
    corpus_ids = subsample(corpus_ids, vocab.freqs, rng=rng)

    # Build negative-sampling table
    print("\n" + "=" * 60)
    print(" STEP 4 / 5 - Building negative-sampling table")
    print("=" * 60)
    neg_table = build_neg_table(vocab)

    # Initialise model & train
    print("\n" + "=" * 60)
    print(" STEP 5 / 5 - Training Skip-Gram with Negative Sampling")
    print("=" * 60)
    model = SkipGramNS(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        seed=config.SEED,
    )
    model = train(model, corpus_ids, neg_table, vocab)

    # Evaluate
    print("\n")
    embeddings = model.get_embeddings()
    nearest_neighbours(embeddings, vocab, top_k=config.TOP_K_NEIGHBOURS)
    word_analogies(embeddings, vocab, top_k=config.TOP_K_ANALOGIES)

    print("Done.")


if __name__ == "__main__":
    main()
