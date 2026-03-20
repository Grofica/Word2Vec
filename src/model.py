import numpy as np


def initialize(V, N) -> tuple:
    """
    :param V: number of words
    :param N: number of dimensions
    :return: tuple of W and W_prime matrix
    """
    np.random.seed(42)
    W = np.random.randn(V, N) / np.sqrt(N)   # (V, N)
    W_prime = np.random.randn(N, V) / np.sqrt(N)   # (N, V)
    return W, W_prime


def get_h(target_idx, W) -> np.ndarray:
    """
    :param target_idx: integer index of target word
    :param W: input weight matrix (V, N)
    :return: h vector, matching row from W matrix

    Equation from paper Johnson et al. - between eq. 26 and eq. 27:
    h = W^T(k,.) = vwI^T
    """
    return W[target_idx]                            # shape: (N,)


def get_u(h, ind, W_prime) -> float:
    """
    :param h: hidden layer vector (N,)
    :param ind: index of word to score
    :param W_prime: output weight matrix (N, V)
    :return: score for word at index ind

    Equation 2 from paper  Johnson et al.:
    uc,j = v'wj^T * h
    """
    return h @ W_prime[:, ind]                      # single score


def get_negative_samples(target_idx, context_idx,
                          V, word_probs, k=5) -> np.ndarray:
    """
    :param target_idx: index of target word
    :param context_idx: index of context word
    :param V: vocabulary size
    :param word_probs: sampling probabilities for each word
    :param k: how many negative samples to return
    :return: array of negative sample indices

    Sample k words that are NOT the target or context.
    More frequent words have a higher chance of being selected.

    Formula from Mikolov et al. 2013:
    P(wi) = f(wi)^(3/4) / sum(f(wj)^(3/4))

    """
    samples = np.random.choice(V, size=k + 10, p=word_probs)
    neg_samples = [s for s in samples
                   if s != target_idx and s != context_idx]
    return np.array(neg_samples[:k])


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def train_step(target_idx, context_idx,
               W, W_prime,
               word_probs,
               learning_rate=0.01,
               k=5) -> float:
    """
    Single training step for Skip-gram with Negative Sampling.

    :param target_idx: index of TARGET word (input)
    :param context_idx: index of CONTEXT word (output to predict)
    :param W: input weight matrix (V, N)
    :param W_prime: output weight matrix (N, V)
    :param word_probs: sampling probabilities for negative sampling
    :param learning_rate: learning rate η (adaptive, passed from train loop)
    :param k: number of negative samples
    :return: loss for this step
    """
    V = W.shape[0]

    # Equation 27 — hidden layer
    h = get_h(target_idx, W)

    # negative samples — vectorized
    neg_samples = get_negative_samples(target_idx, context_idx,
                                       V, word_probs, k)

    #POSITIVE SAMPLE
    u_pos = get_u(h, context_idx, W_prime)
    sig_pos = sigmoid(u_pos)
    L_pos = -np.log(sig_pos + 1e-10)

    #NEGATIVE SAMPLES
    U_neg = W_prime[:, neg_samples]               # (N, k)
    u_negs = U_neg.T @ h                          # (k,)
    sig_negs = sigmoid(u_negs)                       # (k,)
    L_neg = -np.sum(np.log(1 - sig_negs + 1e-10))

    #GRADIENTS

    # gradient from positive sample
    grad_h = (sig_pos - 1) * W_prime[:, context_idx].copy()  # (N,)

    # gradient from negative samples — vectorized
    grad_h += U_neg @ sig_negs                       # (N,)

    #UPDATE W'
    # positive sample — target=1, error = sig_pos - 1
    W_prime[:, context_idx] -= learning_rate * (sig_pos - 1) * h


    W_prime[:, neg_samples] -= learning_rate * np.outer(h, sig_negs)

    #UPDATE W
    W[target_idx] -= learning_rate * grad_h

    return L_pos + L_neg


def most_similar(word, word2idx, idx2word, W, top_n=5) -> None:
    """
    :param word: target word
    :param word2idx: dict -> word:index
    :param idx2word: dict -> index:word
    :param W: input weight matrix (V, N)
    :param top_n: number of nearest neighbors
    """
    if word not in word2idx:
        print(f"Word '{word}' not found in vocabulary!")
        return

    idx = word2idx[word]
    h = get_h(idx, W)
    norms = np.linalg.norm(W, axis=1)
    sims = W @ h / (norms * np.linalg.norm(h) + 1e-10)

    top_idxs = np.argsort(sims)[::-1][1:top_n + 1]

    print(f"\nMost similar words to '{word}':")
    for i in top_idxs:
        print(f"  {idx2word[i]:<15} similarity: {sims[i]:.4f}")