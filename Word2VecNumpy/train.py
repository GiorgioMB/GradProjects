"""
train.py - Training loop for Skip-Gram with Negative Sampling.
"""

from __future__ import annotations

import time

import numpy as np

import config
from data import Vocabulary, generate_batches
from model import SkipGramNS
from utils import save_embeddings


def _linear_lr(
    initial_lr: float,
    min_lr: float,
    progress: float,
) -> float:
    """Linearly decay the learning rate from ``initial_lr`` to ``min_lr``.

    Parameters
    ----------
    initial_lr : float
        Starting learning rate.
    min_lr : float
        Floor - LR will not drop below this.
    progress : float
        Fraction of total training completed, in [0, 1].

    Returns
    -------
    float
        Current learning rate.
    """
    return max(min_lr, initial_lr * (1.0 - progress))


def train(
    model: SkipGramNS,
    corpus_ids: np.ndarray,
    neg_table: np.ndarray,
    vocab: Vocabulary,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    min_lr: float = config.MIN_LR,
    window: int = config.WINDOW_SIZE,
    num_neg: int = config.NUM_NEGATIVES,
    batch_size: int = config.BATCH_SIZE,
    log_every: int = config.LOG_EVERY,
    seed: int = config.SEED,
) -> SkipGramNS:
    """Train the :class:`SkipGramNS` model for ``epochs`` full passes.

    Parameters
    ----------
    model : SkipGramNS
        The model whose parameters will be updated in-place.
    corpus_ids : np.ndarray
        Subsampled, ID-encoded corpus.
    neg_table : np.ndarray
        Pre-built unigram table for negative sampling.
    vocab : Vocabulary
        Vocabulary object (used only for saving).
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate.
    min_lr : float
        Minimum learning rate (linear decay floor).
    window : int
        Maximum context window.
    num_neg : int
        Number of negatives per positive pair.
    batch_size : int
        Mini-batch size.
    log_every : int
        Print progress every this many steps.
    seed : int
        Random seed.

    Returns
    -------
    SkipGramNS
        The trained model (same object, updated in-place).
    """
        rng = np.random.default_rng(seed)

    # Initial estimate for total steps (refined online after each epoch).
    est_pairs_per_epoch = len(corpus_ids) * (window + 1)
    estimated_total_steps = max((est_pairs_per_epoch * epochs) // batch_size, 1)

    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        # Re-seed the batch generator each epoch so window sampling
        # varies, but remains reproducible.
        batch_rng = np.random.default_rng(rng.integers(0, 2**31))
        batches = generate_batches(
            corpus_ids,
            neg_table,
            window=window,
            num_neg=num_neg,
            batch_size=batch_size,
            rng=batch_rng,
        )

        for centers, contexts, negatives in batches:
            progress = global_step / max(estimated_total_steps, 1)
            current_lr = _linear_lr(lr, min_lr, progress)

            batch_loss = model.train_step(
                centers, contexts, negatives, current_lr
            )

            epoch_loss += batch_loss
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                print(
                    f"  [epoch {epoch}/{epochs}]  "
                    f"step {global_step:>8,}  "
                    f"lr {current_lr:.5f}  "
                    f"loss {avg:.4f}  "
                    f"elapsed {elapsed:.0f}s"
                )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch} done "
            f"avg loss {avg_epoch_loss:.4f}, "
            f"{elapsed:.0f}s total"
        )

        observed_avg_steps = global_step / epoch
        estimated_total_steps = max(int(observed_avg_steps * epochs), 1)

    save_embeddings(model, vocab)

    total_time = time.time() - t0
    print(f"\n[train] Training complete in {total_time:.0f}s")
    return model
