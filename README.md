# Word2Vec — Skip-gram with Negative Sampling

Implementation of Word2Vec in pure NumPy (no PyTorch / TensorFlow).

---

## References

This implementation is based on the following papers:

**[1] Mikolov et al. (2013a)** — *Efficient Estimation of Word Representations in Vector Space*
- Skip-gram and CBOW model architectures
- One-word and multi-word context formulations
- Forward pass and gradient derivations

**[2] Mikolov et al. (2013b)** — *Distributed Representations of Words and Phrases and their Compositionality*
- Negative Sampling as an alternative to full Softmax
- Subsampling of frequent words
- Linearly decaying learning rate

**[3] Ayyadevara, V. K. (2018)** — *Pro Machine Learning Algorithms*, Chapter 8: Word2Vec. Apress.
- Step-by-step derivation of the forward pass
- Cross-entropy loss and gradient derivations
- Hidden layer weight matrix as word vectors

---

## Model

Skip-gram with Negative Sampling:

- **Input**: one target word (one-hot encoded)
- **Hidden layer**: linear, no activation — row of W matrix
- **Output**: probability distribution over context words
- **Optimization**: Negative Sampling instead of full Softmax

### Key equations

**Hidden layer** — equation between eq. 26 and eq. 27 in [1]:
```
h = W[target_idx, :]
```

**Output scores** — equation 2 in [1]:
```
u[j] = v'_wj^T * h
```

**Negative Sampling loss** — [2]:
```
L = -log σ(v'_context · h) - Σk log σ(-v'_neg · h)
```

**Negative sampling probabilities** — [2]:
```
P(wi) = f(wi)^(3/4) / Σ f(wj)^(3/4)
```

**Linearly decaying learning rate** — [2]:
```
lr = lr_start * (1 - current_step / total_steps)
lr = max(lr, lr_start * 0.0001)
```

---

## Project structure

```
word2vec/
├── README.md
├── main.py                ← training entry point
├── demo.ipynb             ← word similarity, vector arithmetic, PCA visualization
├── .gitignore
└── src/
    ├── __init__.py
    ├── preprocess.py      ← tokenization, vocabulary, skip-gram pairs
    ├── model.py           ← forward pass, gradients, parameter updates
    └── train.py           ← training loop, negative sampling probabilities
```

---

## Setup

```bash
pip install numpy safetensors scipy matplotlib
```

Download dataset:
```bash
mkdir data
wget http://mattmahoney.net/dc/text8.zip -O data/text8.zip
unzip data/text8.zip -d data/
```

---

## Training

```bash
python main.py
```

Hyperparameters in `main.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WORDS_SUBSET` | 1,000,000 | Number of words used for training |
| `MIN_COUNT` | 50 | Minimum word frequency [3] |
| `WINDOW_SIZE` | 2 | Context window size [1] |
| `N` | 100 | Embedding dimension |
| `LEARNING_RATE` | 0.025 | Initial learning rate [2] |
| `K` | 5 | Number of negative samples [2] |
| `EPOCHS` | 3 | Number of training epochs |

---

## Evaluation

Model is evaluated on:

- **WordSim353** — word similarity benchmark, measured with Spearman correlation (Table 4 in [1])
- **Analogy task** — semantic and syntactic analogies, measured with accuracy (Table 5 and 6 in [1])

Both evaluations are available in `demo.ipynb`.
