"""
config.py - Every parameter lives here.
"""

EMBED_DIM: int = 100

MAX_VOCAB_SIZE: int = 20_000
"""Keep only the top-N most frequent words."""

MIN_COUNT: int = 5
"""Discard words that appear fewer than this many times in the corpus."""

SUBSAMPLE_THRESH: float = 1e-3
"""Subsampling threshold t, following Mikolov et al. (2013), 
each word w with frequency f(w) is kept with probability 
P(w) = \min(\max(0, 1 - \sqrt(\frac{t}{f(w)})), 1)"""

WINDOW_SIZE: int = 5
"""Maximum context window on each side.  For each training example the
actual window is sampled uniformly from [1, WINDOW_SIZE]."""

NUM_NEGATIVES: int = 5
"""Number of negative samples drawn per (center, context) positive pair."""

EPOCHS: int = 5
"""Number of full passes through the corpus."""

LEARNING_RATE: float = 0.025
"""Initial learning rate for SGD."""

MIN_LR: float = 1e-4
"""Learning rate will not decay below this value."""

BATCH_SIZE: int = 256
"""Number of (center, context) pairs per mini-batch SGD step."""

NEG_TABLE_SIZE: int = 10_000_000
"""Size of the unigram table for {\cal O}(1) negative sampling."""

NEG_POWER: float = 0.75
"""Exponent applied to unigram frequencies when building the negative-
sampling distribution.  0.75 is the value used by the original C code."""

DATA_DIR: str = "data"
EMBED_DIR: str = "embeddings"


SEED: int = 42
LOG_EVERY: int = 10_000
