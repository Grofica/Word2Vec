# Word2Vec — Skip-gram with Negative Sampling

Implementation of Word2Vec only in NumPy.

---

## References

This implementation is based on the following papers:

**[1] Mikolov et al. (2013a)** — *Efficient Estimation of Word Representations in Vector Space*

**[2] Mikolov et al. (2013b)** — *Distributed Representations of Words and Phrases and their Compositionality*


**[3] Ayyadevara, V. K. (2018)** — *Pro Machine Learning Algorithms*, Chapter 8: Word2Vec. Apress.

---

## Model

Skip-gram with Negative Sampling:

- **Input**: one target word (one-hot encoded)
- **Hidden layer**: linear, no activation — row of W matrix
- **Output**: probability distribution over context words
- **Optimization**: Negative Sampling

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

## Training

| Epoch | Avg Loss | Learning Rate |
|-------|----------|---------------|
| 1     | 2.2564   | 0.016687      |
| 2     | 2.1866   | 0.008354      |
| 3     | 2.1191   | 0.000021      |

Dataset: 1,000,000 words | Epochs: 3 | N: 100 | Window: 2 | K: 5
