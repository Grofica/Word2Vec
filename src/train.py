import random
import numpy as np
from src.model import train_step


def build_word_probs(word_counts, idx2word, V) -> np.ndarray:
    """
    Negative sampling probability formula from Mikolov et al. 2013:
    P(wi) = f(wi)^(3/4) / sum(f(wj)^(3/4))

    """
    word_freqs  = np.array([word_counts[idx2word[i]] for i in range(V)])
    word_freqs_powered = word_freqs ** 0.75
    word_probs = word_freqs_powered / word_freqs_powered.sum()
    return word_probs


def train(pairs, W, W_prime,
          word_probs,
          epochs=3,
          learning_rate=0.025,
          k=5,
          log_every=10_000) -> tuple:
    """
    Training loop for Skip-gram with Negative Sampling.

    :param pairs: list of (target_idx, context_idx) pairs
    :param W: input weight matrix (V, N)
    :param W_prime: output weight matrix (N, V)
    :param word_probs: probabilities for negative sampling
    :param epochs: number of training epochs
    :param learning_rate: initial learning rate η
    :param k: number of negative samples
    :param log_every: how often to print the loss
    :return: W, W_prime, list of average losses

    Adaptive learning rate was implemented 'cause this paper:
     "Efficient Estimation of Word Representations in Vector Space"
     by  Mikolov et. al (2013)
    """
    losses = []
    total_steps = len(pairs) * epochs

    for epoch in range(epochs):
        total_loss = 0

        # shuffle pairs at the start of each epoch
        random.shuffle(pairs)

        for i, (target_idx, context_idx) in enumerate(pairs):

            # Adaptive learning rate (Mikolov et al. 2013)
            current_step = epoch * len(pairs) + i
            progress = current_step / total_steps
            current_lr = learning_rate * (1 - progress)
            current_lr = max(current_lr, learning_rate * 0.0001)

            # single training step
            L = train_step(target_idx, context_idx,
                           W, W_prime,
                           word_probs,
                           current_lr, k)
            total_loss += L

            # logging
            if (i + 1) % log_every == 0:
                avg_loss  = total_loss / log_every
                current_step_global = epoch * len(pairs) + (i + 1)
                losses.append(avg_loss)
                print(f"Ep {epoch + 1}/{epochs} | "
                      f"St {current_step_global}/{total_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.6f}")
                total_loss = 0

        print(f"\nEpoch {epoch + 1} complete!\n")

    return W, W_prime, losses