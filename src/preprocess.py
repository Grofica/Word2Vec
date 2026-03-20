import re
from collections import Counter


def load_text(filepath) -> str:
    """
    :param filepath: PATH to the text file
    Load text from filepath
    """
    with open(filepath, 'r') as f:
        return f.read()


def tokenize(text) -> list[str]:
    """
    Clean and tokenize text:
    - lowercase everything
    - remove everything except letters and spaces
    - split into words
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


def build_vocab(words, min_count=50):
    """
    :param words: list of words
    :param min_count: minimum number of times a word appears
    :return: word2idx, idx2word, filtered words, word_counts
    This was done in Ayyadavera work
    """
    word_counts = Counter(words)


    word_counts = {
        word: count
        for word, count in word_counts.items()
        if count >= min_count
    }

    word2idx = {word: idx for idx, word in enumerate(word_counts.keys())}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # keep only words that are in vocabulary
    words = [word for word in words if word in word2idx]

    return word2idx, idx2word, words, word_counts


def generate_skipgram_pairs(words, word2idx, window_size=2) -> list[tuple]:
    """
    :param words: list of words
    :param word2idx: dict - word:index
    :param window_size: int range to look around target word
    :return: pairs of (target_idx, context_idx)

    For each target word, generate a pair with every
    context word within the window_size.

    Example (window_size=2):
    "the quick brown fox jumped"
            ^
          target = "brown"
    pairs: ("brown","the"), ("brown","quick"),
           ("brown","fox"), ("brown","jumped")
    """
    pairs = []

    for i, target_word in enumerate(words):
        target_idx = word2idx[target_word]

        for j in range(-window_size, window_size + 1):
            if j == 0:
                continue

            context_pos = i + j

            if context_pos < 0 or context_pos >= len(words):
                continue

            context_idx = word2idx[words[context_pos]]
            pairs.append((target_idx, context_idx))

    return pairs