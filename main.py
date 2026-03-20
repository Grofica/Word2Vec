from src.preprocess import load_text, tokenize, build_vocab, generate_skipgram_pairs
from src.model import initialize, most_similar
from src.train import build_word_probs, train
from safetensors.numpy import save_file
import json

#  Hyperparameters
DATA_PATH  = 'data/text8'
MIN_COUNT = 50            # minimum word frequency
WINDOW_SIZE = 2           # context window size (left and right)
N  = 100                  # embedding dimension
LEARNING_RATE = 0.025     # initial learning rate η (Mikolov et al. 2013)
K = 5                     # number of negative samples
EPOCHS = 3                 # number of training epochs
WORDS_SUBSET = 1_000_000   # number of words to use for training


def main():

    #Load and tokenize
    print("=" * 50)
    print("Load and tokenize")
    print("=" * 50)

    text  = load_text(DATA_PATH)
    words = tokenize(text)
    print(f"Total words: {len(words)}")

    #Build vocabulary
    print("\n" + "=" * 50)
    print("Build vocabulary")
    print("=" * 50)

    word2idx, idx2word, words, word_counts = build_vocab(words, MIN_COUNT)
    V = len(word2idx)

    print(f"Vocabulary size V: {V}")
    print(f"Total words after filtering: {len(words)}")

    #Generate Skip-gram pairs
    print("\n" + "=" * 50)
    print("Generate Skip-gram pairs")
    print("=" * 50)

    words_subset = words[:WORDS_SUBSET]
    pairs        = generate_skipgram_pairs(words_subset, word2idx, WINDOW_SIZE)

    print(f"Total pairs: {len(pairs)}")
    print(f"\nSample pairs (target, context):")
    for i in range(5):
        tgt, ctx = pairs[i]
        print(f"  target: '{idx2word[tgt]}' -> context: '{idx2word[ctx]}'")

    #Initialize model
    print("\n" + "=" * 50)
    print("Initialize model")
    print("=" * 50)

    W, W_prime = initialize(V, N)

    print(f"W shape:       {W.shape}")
    print(f"W_prime shape: {W_prime.shape}")

    #Build negative sampling probabilities
    print("\n" + "=" * 50)
    print("Build negative sampling probabilities")
    print("=" * 50)

    word_probs = build_word_probs(word_counts, idx2word, V)
    print(f"Sum of probabilities: {word_probs.sum():.4f}")

    #Train
    print("\n" + "=" * 50)
    print("Train")
    print("=" * 50)

    W, W_prime, losses = train(pairs=pairs,
                               W=W,
                               W_prime=W_prime,
                               word_probs=word_probs,
                               epochs=EPOCHS,
                               learning_rate=LEARNING_RATE,
                               k=K,               # BUG FIX: n_negs → k
                               log_every=10_000)

    #Save model
    tensors = {"W": W, "W_prime": W_prime}
    save_file(tensors, "model.safetensors")
    print("Model saved to model.safetensors")
    vocab = {
        "word2idx": word2idx,
        "idx2word": {str(k): v for k, v in idx2word.items()}
    }
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

    print("Vocab saved to vocab.json")

    #Evaluation
    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)

    test_words = ['king', 'paris', 'woman', 'computer', 'the']
    for word in test_words:
        most_similar(word, word2idx, idx2word, W, top_n=5)


if __name__ == '__main__':
    main()