"""
evaluate.py - Qualitative evaluation of learned word embeddings.

Two evaluation modes are provided:

1. Nearest neighbours - for a list of query words, print the top-k
   most similar words by cosine similarity.
2. Word analogies - test ``A : B :: C : ?`` relationships
   (e.g. king - man + woman \approx queen).
"""

from __future__ import annotations

import numpy as np

from data import Vocabulary



def _normalise_rows(M: np.ndarray) -> np.ndarray:
    """L2-normalise each row of M in-place (safe against zero rows).

    Parameters
    ----------
    M : np.ndarray, shape ``(N, D)``

    Returns
    -------
    np.ndarray, shape ``(N, D)``
        Row-normalised copy.
    """
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return M / norms


# A curated list of common words likely present in WikiText-2.
DEFAULT_QUERIES = [
    "king", "queen", "man", "woman",
    "computer", "university", "city",
    "good", "water", "time",
]


def nearest_neighbours(
    embeddings: np.ndarray,
    vocab: Vocabulary,
    queries: list[str] | None = None,
    top_k: int = 10,
) -> None:
    """Print the top_k nearest neighbours for each query word.

    Similarity is measured by cosine similarity (dot product of
    L2-normalised vectors).

    Parameters
    ----------
    embeddings : np.ndarray, shape ``(V, D)``
        Word embedding matrix.
    vocab : Vocabulary
    queries : list[str], optional
        Words to look up.  Defaults to :data:`DEFAULT_QUERIES`.
    top_k : int
        Number of neighbours to display.
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    normed = _normalise_rows(embeddings)

    print("\n" + "=" * 60)
    print(" NEAREST NEIGHBOURS (cosine similarity)")
    print("=" * 60)

    for word in queries:
        if word not in vocab:
            print(f"  '{word}' not in vocabulary — skipping")
            continue

        wid = vocab.word2id[word]
        vec = normed[wid]                     # (D,)
        sims = normed @ vec                    # (V,)

        # Exclude the query word itself
        sims[wid] = -1.0
        top_ids = np.argsort(sims)[::-1][:top_k]

        neighbours = [
            f"{vocab.id2word[i]} ({sims[i]:.3f})" for i in top_ids
        ]
        print(f"\n  {word}:")
        for nb in neighbours:
            print(f"    {nb}")

    print()

# Analogy tuples: (A, B, C, expected D) such that A:B :: C:D
DEFAULT_ANALOGIES = [
    ("king", "man", "queen", "woman"),
    ("paris", "france", "london", "england"),
    ("big", "bigger", "small", "smaller"),
    ("man", "woman", "boy", "girl"),
    ("good", "best", "bad", "worst"),
]


def word_analogies(
    embeddings: np.ndarray,
    vocab: Vocabulary,
    analogies: list[tuple[str, str, str, str]] | None = None,
    top_k: int = 5,
) -> None:
    """Evaluate word-analogy accuracy.

    For each analogy ``(A, B, C, D)``, compute

        v = emb(A) - emb(B) + emb(C)

    and check whether ``D`` appears among the *top_k* nearest words
    to ``v`` (excluding A, B, C).

    Parameters
    ----------
    embeddings : np.ndarray, shape ``(V, D)``
    vocab : Vocabulary
    analogies : list of 4-tuples, optional
    top_k : int
        How many candidates to inspect.
    """
    if analogies is None:
        analogies = DEFAULT_ANALOGIES

    normed = _normalise_rows(embeddings)

    print("=" * 60)
    print(" WORD ANALOGIES  (A - B + C ≈ ?)")
    print("=" * 60)

    hits = 0
    total = 0

    for a, b, c, expected in analogies:
        # Skip if any word is OOV
        if not all(w in vocab for w in (a, b, c, expected)):
            missing = [w for w in (a, b, c, expected) if w not in vocab]
            print(f"  {a}:{b} :: {c}:? — skipped (OOV: {missing})")
            continue

        total += 1
        ids = [vocab.word2id[w] for w in (a, b, c)]
        vec = normed[ids[0]] - normed[ids[1]] + normed[ids[2]]

        # Normalise the query
        vec /= max(np.linalg.norm(vec), 1e-12)
        sims = normed @ vec

        # Exclude the three input words
        for idx in ids:
            sims[idx] = -1.0

        top_ids = np.argsort(sims)[::-1][:top_k]
        top_words = [vocab.id2word[i] for i in top_ids]

        found = expected in top_words
        hits += int(found)
        mark = "✓" if found else "✗"
        print(
            f"  {mark}  {a} - {b} + {c} = "
            f"{top_words[0]}  (expected: {expected})  "
            f"top-{top_k}: {top_words}"
        )

    if total > 0:
        print(f"\n  Accuracy: {hits}/{total} = {100 * hits / total:.1f}%")
    else:
        print("  (no analogies could be evaluated)")
    print()
