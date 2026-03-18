"""
data.py - Corpus loading, vocabulary construction, subsampling,
negative-sampling table, and training-pair generation.

Pipeline:
load_corpus         -> Download / read WikiText-2 and tokenise.
build_vocab         -> Frequency counts and word <--> id mappings.
subsample           -> Probabilistically discard frequent words from the corpus.
build_neg_table     -> Pre-compute a large table for {\cal O}(1) neg sampling.
generate_batches    -> Yield mini-batches of (center, context, negatives).
"""
from __future__ import annotations
import os
import re
import shutil
import urllib.request
from urllib.error import HTTPError, URLError
import zipfile
from collections import Counter
from typing import Generator
import numpy as np
import config

_WIKITEXT2_URLS = (
    "https://wikitext.smerity.com/wikitext-2-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
)


def _download_file(url: str, output_path: str) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "word2vec-training-script/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response, open(
        output_path, "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)

def _download_wikitext2(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "wikitext-2-v1.zip")
    extract_dir = os.path.join(data_dir, "wikitext-2")
    train_path = os.path.join(extract_dir, "wiki.train.tokens")

    if os.path.isfile(train_path):
        return train_path

    if not os.path.isfile(zip_path) or not zipfile.is_zipfile(zip_path):
        if os.path.isfile(zip_path):
            print(f"[data] Removing invalid archive: {zip_path}")
            os.remove(zip_path)

        print(f"[data] Downloading WikiText-2, {zip_path}")
        last_error: Exception | None = None
        for url in _WIKITEXT2_URLS:
            try:
                print(f"[data] Trying: {url}")
                _download_file(url, zip_path)
                if not zipfile.is_zipfile(zip_path):
                    raise zipfile.BadZipFile("Downloaded file is not a valid zip")
                break
            except (HTTPError, URLError, TimeoutError, zipfile.BadZipFile) as err:
                last_error = err
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
        else:
            raise RuntimeError(
                "Failed to download WikiText-2 from all known URLs"
            ) from last_error

    print(f"[data] Extracting, {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return train_path


def _tokenise(text: str) -> list[str]:
    """Lower-case and split on whitespace / punctuation, keeping only
    alphabetic tokens of length ≥ 2."""
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return tokens


def load_corpus(data_dir: str = config.DATA_DIR) -> list[str]:
    train_path = _download_wikitext2(data_dir)
    with open(train_path, encoding="utf-8") as fh:
        raw = fh.read()
    tokens = _tokenise(raw)
    print(f"[data] Corpus loaded: {len(tokens):,} tokens")
    return tokens


class Vocabulary:
    """Word <--> integer-ID mapping with frequency information.

    Attributes
    ----------
    word2id : dict[str, int] 
        Mapping from word string to integer index.
    id2word : dict[int, str] 
        Reverse mapping.
    counts  : np.ndarray, shape (vocab_size,) 
        Raw frequency of each word in the corpus.
    freqs   : np.ndarray, shape (vocab_size,) 
        Normalised frequency (sums to 1).
    """

    def __init__(
        self,
        word2id: dict[str, int],
        id2word: dict[int, str],
        counts: np.ndarray,
    ):
        self.word2id = word2id
        self.id2word = id2word
        self.counts = counts
        self.freqs: np.ndarray = counts / counts.sum()

    def __len__(self) -> int:
        return len(self.word2id)

    def __contains__(self, word: str) -> bool:
        return word in self.word2id


def build_vocab(
    tokens: list[str],
    max_vocab: int = config.MAX_VOCAB_SIZE,
    min_count: int = config.MIN_COUNT,
) -> Vocabulary:
    counter = Counter(tokens)
    # Filter by min_count and take the top-N
    most_common = [
        (w, c) for w, c in counter.most_common(max_vocab) if c >= min_count
    ]
    word2id: dict[str, int] = {}
    id2word: dict[int, str] = {}
    counts_list: list[int] = []

    for idx, (word, count) in enumerate(most_common):
        word2id[word] = idx
        id2word[idx] = word
        counts_list.append(count)

    counts = np.array(counts_list, dtype=np.float64)
    vocab = Vocabulary(word2id, id2word, counts)
    print(f"[data] Vocabulary built: {len(vocab):,} words")
    return vocab

def subsample(
    token_ids: np.ndarray,
    freqs: np.ndarray,
    threshold: float = config.SUBSAMPLE_THRESH,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Probabilistically discard frequent words (Mikolov et al., 2013).

    Parameters
    ----------
    token_ids : np.ndarray, shape ``(N,)``
        Integer-encoded corpus.
    freqs : np.ndarray, shape ``(V,)``
        Normalised word frequencies.
    threshold : float
        Subsampling threshold t.
    rng : np.random.Generator, optional
        Random number generator (for reproducibility).

    Returns
    -------
    np.ndarray
        Filtered array of token IDs (shorter than input).
    """
    if rng is None:
        rng = np.random.default_rng(config.SEED)

    word_freqs = freqs[token_ids]

    keep_prob = 1.0 - np.sqrt(threshold / word_freqs)
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    # We keep a token when a uniform draw exceeds keep_prob
    # (note: higher keep_prob -> more likely to DROP)
    mask = rng.random(len(token_ids)) > keep_prob
    subsampled = token_ids[mask]
    print(
        f"[data] Subsampling: {len(token_ids):,} -> {len(subsampled):,} tokens "
        f"({100 * len(subsampled) / len(token_ids):.1f}% kept)"
    )
    return subsampled


def build_neg_table(
    vocab: Vocabulary,
    table_size: int = config.NEG_TABLE_SIZE,
    power: float = config.NEG_POWER,
) -> np.ndarray:
    r"""Pre-compute a unigram table for $O(1)$ negative sampling.

    Each entry in the returned array is a word ID.  Drawing uniformly
    from this table is equivalent to sampling from the smoothed unigram
    distribution 
    
    $$
    P(w) \propto \text{count}(w)^{\text{power}}
    $$

    The table approach is the same trick used in the original C
    implementation of Word2Vec.

    Parameters
    ----------
    vocab : Vocabulary
    table_size : int
        Number of entries in the table.
    power : float
        Exponent applied to raw counts (typically 0.75).

    Returns
    -------
    np.ndarray, shape ``(table_size,)``, dtype int64
    """
    smoothed = vocab.counts ** power
    smoothed /= smoothed.sum()  # normalise

    table = np.zeros(table_size, dtype=np.int64)
    cumulative = 0.0
    word_id = 0
    for i in range(table_size):
        table[i] = word_id
        # Advance to the next word when we've filled its share of the table
        if (i + 1) / table_size > cumulative + smoothed[word_id]:
            cumulative += smoothed[word_id]
            word_id = min(word_id + 1, len(vocab) - 1)

    print(f"[data] Negative-sampling table built ({table_size:,} entries)")
    return table


def generate_batches(
    corpus_ids: np.ndarray,
    neg_table: np.ndarray,
    window: int = config.WINDOW_SIZE,
    num_neg: int = config.NUM_NEGATIVES,
    batch_size: int = config.BATCH_SIZE,
    rng: np.random.Generator | None = None,
) -> Generator[
    tuple[np.ndarray, np.ndarray, np.ndarray], None, None
]:
    """Yield mini-batches of Skip-Gram training pairs with negatives.

    For every token in ``corpus_ids`` (the center word), we sample a
    random window size ``w ~ Uniform(1, window)`` and pair the center
    with each context word within that window.  Negative samples are
    drawn from ``neg_table``.

    Yields
    ------
    centers  : np.ndarray, shape ``(B,)``
        Center-word IDs.
    contexts : np.ndarray, shape ``(B,)``
        Positive-context-word IDs.
    negatives : np.ndarray, shape ``(B, num_neg)``
        Negative-sample IDs for each pair.
    """
    if rng is None:
        rng = np.random.default_rng(config.SEED)

    n = len(corpus_ids)
    # Pre-allocate buffers for the current batch
    buf_center = np.empty(batch_size, dtype=np.int64)
    buf_context = np.empty(batch_size, dtype=np.int64)
    buf_neg = np.empty((batch_size, num_neg), dtype=np.int64)
    ptr = 0  # pointer into the batch buffer

    for i in range(n):
        center = corpus_ids[i]
        # Dynamic window, sample w in [1, window]
        w = rng.integers(1, window + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)

        for j in range(start, end):
            if j == i:
                continue
            buf_center[ptr] = center
            buf_context[ptr] = corpus_ids[j]
            # Draw negatives from the table
            neg_indices = rng.integers(0, len(neg_table), size=num_neg)
            buf_neg[ptr] = neg_table[neg_indices]
            ptr += 1

            if ptr == batch_size:
                yield (
                    buf_center.copy(),
                    buf_context.copy(),
                    buf_neg.copy(),
                )
                ptr = 0

    # Flush remaining pairs
    if ptr > 0:
        yield (
            buf_center[:ptr].copy(),
            buf_context[:ptr].copy(),
            buf_neg[:ptr].copy(),
        )
