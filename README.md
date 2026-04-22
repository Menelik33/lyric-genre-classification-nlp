# NLP Lyric Genre Identifier

Predicting a song's genre (Hip-Hop, Pop, or Rock) from its lyrics using an NLP pipeline built with TensorFlow/Keras.

Three models of increasing sophistication are implemented, demonstrating progression from a simple baseline to transfer learning with fine-tuned word embeddings.

---

## Dataset

Song lyrics labeled with genre, sourced from the MIT XPro Lyric Genre dataset.

| Split | File | Samples |
|-------|------|---------|
| Train | `lyric_genre_train.csv` | 48,991 |
| Validation | `lyric_genre_val.csv` | — |
| Test | `lyric_genre_test.csv` | — |

**Genre distribution (train set):** Rock 54.9% · Pop 29.5% · Hip-Hop 15.5%
---

## Pipeline Overview

```
Raw Lyrics
↓
Text Vectorization (Keras TextVectorization)
↓
┌─────────────────────────────────────────────────────┐
│ Model 1         │ Model 2          │ Model 3         │
│ Multi-Hot       │ GloVe Embeddings │ GloVe Embeddings│
│ Encoding        │ (Frozen)         │ (Fine-tuned)    │
└─────────────────────────────────────────────────────┘
↓
GlobalAveragePooling1D (Models 2 & 3)
↓
Feedforward Neural Network
↓
Genre Prediction (softmax, 3 classes)
```

---

---

## Models

### Model 1 — Bag of Words Baseline

Lyrics are vectorized into a 5,000-dimensional multi-hot binary vector. Each position represents a vocabulary word — 1 if present in the lyric, 0 if not. Word order and frequency are discarded.

**Architecture:** `Input(5000)` → `Dense(8, relu)` → `Dense(3, softmax)`

---

### Model 2 — Transfer Learning: Frozen GloVe Embeddings

Pre-trained GloVe vectors (Stanford NLP, 6B tokens, 100 dimensions) are loaded as the embedding layer with `trainable=False`. The model borrows semantic word knowledge without modifying it.

**Architecture:** `Input` → `Embedding(GloVe, frozen)` → `GlobalAveragePooling1D` → `Dense(8)` → `Dropout(0.5)` → `Dense(3, softmax)`

---

### Model 3 — Transfer Learning: Fine-tuned GloVe Embeddings

Same architecture as Model 2, but with `trainable=True` on the embedding layer. Backpropagation updates the GloVe vectors during training, adapting word representations to the lyric domain.

**Architecture:** `Input` → `Embedding(GloVe, fine-tuned)` → `GlobalAveragePooling1D` → `Dense(8)` → `Dropout(0.5)` → `Dense(3, softmax)`

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| Model 1 — Bag of Words Baseline | __% |
| Model 2 — Transfer Learning: Frozen GloVe | __% |
| Model 3 — Transfer Learning: Fine-tuned GloVe | __% |

*Random baseline accuracy: ~33% (3 classes)*

---

## Usage

Designed to run in **Google Colab**.

1. Upload the three CSV files to Google Drive under `MyDrive/XPro/`
2. Open `NLP_lyric_identifier.ipynb` in Colab and mount Drive
3. Run all cells — GloVe embeddings are downloaded automatically
4. Use `lyric_predict()` to classify any lyric fragment:

```python
lyric_predict("I grew up on the crime side, the New York Times side")
# 91.24% Hip-Hop
# 05.43% Pop
# 03.33% Rock
```

---

## Technologies

Python · TensorFlow · Keras · NumPy · Pandas · Matplotlib · GloVe (Stanford NLP)

---

## What This Project Demonstrates

- Building an end-to-end NLP classification pipeline from raw text
- Vocabulary construction and multi-hot feature engineering
- Applying transfer learning with pre-trained word embeddings
- Fine-tuning embeddings to adapt to a specific text domain
- Evaluating model progression across train / validation / test splits
