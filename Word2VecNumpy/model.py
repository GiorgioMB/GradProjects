"""
model.py - Skip-Gram with Negative Sampling (SGNS) model.

Contains the core model logic:

* Embedding matrices (center & context).
* Forward pass - dot-product scores + sigmoid.
* Loss - negative-sampling loss (binary cross-entropy style).
* Analytic gradients - derived by hand (see README for equations).
* SGD parameter update.
"""

from __future__ import annotations
import numpy as np


# Numerically-stable sigmoid
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid with clipping to avoid overflow.

    Uses the identity  f(-x) = 1 - f(x)  via ``np.where`` to handle
    both large positive and large negative inputs stably.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        f(x) \in (0, 1), same shape as input.
    """
    pos_mask = x >= 0
    z = np.zeros_like(x)

    # For x \ge 0:  f(x) = 1 / (1 + exp(-x))
    z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    # For x < 0:  f(x) = exp(x) / (1 + exp(x))   (avoids exp of large +ve)
    exp_x = np.exp(x[~pos_mask])
    z[~pos_mask] = exp_x / (1.0 + exp_x)

    return z


# Model class
class SkipGramNS:
    """Skip-Gram model trained with Negative Sampling.

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary.
    embed_dim : int
        Dimensionality of each word vector.
    seed : int
        Random seed for weight initialisation.

    Attributes
    ----------
    W_center : np.ndarray, shape ``(vocab_size, embed_dim)``
        Center (input) embedding matrix.
    W_context : np.ndarray, shape ``(vocab_size, embed_dim)``
        Context (output) embedding matrix.
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Xavier / Glorot-style initialisation ─ scale ~ 1/sqrt(d)
        scale = 1.0 / np.sqrt(embed_dim)
        self.W_center: np.ndarray = rng.uniform(
            -scale, scale, size=(vocab_size, embed_dim)
        )
        # Context embeddings initialised to zero
        self.W_context: np.ndarray = np.zeros(
            (vocab_size, embed_dim), dtype=np.float64
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def train_step(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        negatives: np.ndarray,
        lr: float,
    ) -> float:
        """Perform one SGD mini-batch update and return the mean loss.

        This method fuses the forward pass, loss computation, gradient
        derivation, and parameter update into a single call for
        efficiency - there is no separate ``backward()`` because we
        never need to build a computation graph.

        Parameters
        ----------
        centers : np.ndarray, shape ``(B,)``
            Center-word IDs for the batch.
        contexts : np.ndarray, shape ``(B,)``
            Positive-context-word IDs.
        negatives : np.ndarray, shape ``(B, K)``
            Negative-sample word IDs (K negatives per pair).
        lr : float
            Current learning rate.

        Returns
        -------
        float
            Mean SGNS loss over the batch.
        """
        B = len(centers)
        K = negatives.shape[1]  # number of negatives

        #Look up embeddings
        v = self.W_center[centers]         # (B, D)
        u_pos = self.W_context[contexts]   # (B, D)
        u_neg = self.W_context[negatives]  # (B, K, D)

        # Forward: compute scores 
        score_pos = np.sum(v * u_pos, axis=1)            # (B,)
        score_neg = np.sum(
            v[:, np.newaxis, :] * u_neg, axis=2
        )                                                 # (B, K)

        # Loss
        sig_pos = _sigmoid(score_pos)                     # (B,)
        sig_neg = _sigmoid(score_neg)                     # (B, K)

        # Numerical safety: clamp away from 0 before taking log
        eps = 1e-7
        loss = (
            -np.log(sig_pos + eps)
            - np.sum(np.log(1.0 - sig_neg + eps), axis=1)
        )                                                 # (B,)
        mean_loss = loss.mean()

        # Gradients
        grad_v = (
            (sig_pos - 1.0)[:, np.newaxis] * u_pos        # (B, D)
            + np.sum(
                sig_neg[:, :, np.newaxis] * u_neg, axis=1  # (B, D)
            )
        )

        grad_u_pos = (sig_pos - 1.0)[:, np.newaxis] * v   # (B, D)

        grad_u_neg = sig_neg[:, :, np.newaxis] * v[:, np.newaxis, :]

        # SGD update 
        self.W_center[centers] -= lr * grad_v

        np.add.at(self.W_context, contexts, -lr * grad_u_pos)

        # Flatten negatives for scatter-add
        neg_flat = negatives.reshape(-1)                   # (B*K,)
        grad_neg_flat = grad_u_neg.reshape(-1, self.embed_dim)
        np.add.at(self.W_context, neg_flat, -lr * grad_neg_flat)

        return float(mean_loss)


    def get_embeddings(self) -> np.ndarray:
        """Return the final word embeddings.

        Following common practice, we return only the center embedding
        matrix ``W_center``.
        Returns
        -------
        np.ndarray, shape ``(vocab_size, embed_dim)``
        """
        return self.W_center.copy()
