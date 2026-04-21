# NLP Lyric Genre Identifier

Predicting a song's genre (Hip-Hop, Pop, or Rock) from its lyrics using an NLP pipeline built with TensorFlow/Keras.

Three models of increasing sophistication are implemented, demonstrating progression from a simple baseline to transfer learning with fine-tuned word embeddings.

Example input:

"I'm standing on the corner with my guitar tonight"

Predicted output:

Pop

---

## Dataset

The datasets contain song lyrics labeled with their corresponding genre.

| Column | Description |
|------|------|
| Lyric | Song lyric text |
| Genre | Target genre label |

Example:

| Lyric | Genre |
|------|------|
| "I got the horses in the back..." | Country |
| "We will rock you..." | Rock |
| "Drop it like it's hot..." | Hip-Hop |

There are 3 datasets:
• Training set  
• Validation set  
• Test set  

This allows the model to be trained and evaluated on unseen data.

---

## System Architecture

The pipeline follows the following structure:

Raw Lyrics  
↓  
Text Vectorization  
↓  
Multi-Hot Feature Representation  
↓  
Feedforward Neural Network  
↓  
Genre Prediction  

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
## 5 Results

The model was evaluated on a seperate test dataset using model.evaluate().

With three possible genres, a random baseline accuracy would be approximately 33%.

The trained model achieved:

Test Accuracy: ~72%
Test Loss: ~0.82 (categorical crossentropy)
681/681 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.7222 - loss: 0.8161

This demonstrates the effectiveness of embedding-based representations over simple statistical baselines for text classification tasks.

## Training Process

The model is trained using:

• Cross-entropy loss  
• Backpropagation  
• Gradient descent optimization  

Training data is used to learn patterns in lyrics associated with different genres.

Validation data is used to monitor generalization performance.

Test data is used for final evaluation.

---

## Key NLP Techniques Used

• Text vectorization using **Keras TextVectorization**  
• Vocabulary construction from raw text  
• Multi-hot encoding for high-dimensional sparse text representation  
• Genre label encoding via one-hot vectors  
• Token frequency-based vocabulary selection  
• Neural network text classification  

---

## Machine Learning Techniques

• Feedforward neural networks  
• Softmax multi-class classification  
• Cross-entropy optimization  
• Train / validation / test dataset splitting  
• Model evaluation on unseen data  

---

## Technologies Used

• Python  
• TensorFlow  
• Keras  
• NumPy  
• Pandas  
• Matplotlib  

---

## What This Project Demonstrates

This project demonstrates the ability to:

• Build NLP preprocessing pipelines for text data  
• Transform raw language into numerical ML features  
• Construct vocabularies and vectorization layers  
• Train neural networks for text classification  
• Evaluate models using structured dataset splits  
• Work with high-dimensional sparse text representations  

---

## Key Technical Contributions

### Custom Text Vectorization Pipeline

Implemented a preprocessing pipeline using **Keras TextVectorization** to convert raw lyrics into numerical feature vectors suitable for neural network training.

### Vocabulary Construction from Training Corpus

Built a 5,000-token vocabulary from the training dataset by scanning word frequencies and mapping each token to a unique index.

### Sparse Multi-Hot Feature Engineering

Represented lyrics as **multi-hot vectors**, enabling efficient learning from high-dimensional sparse text features.

### Neural Genre Classification Model

Designed and trained a neural network capable of predicting music genre based on textual patterns within song lyrics.

### Structured Training Workflow

Implemented training, validation, and test dataset separation to properly evaluate model performance on unseen data.

---

## Example Prediction

Input lyric:

"I will party tonight until the sun comes up"

Model prediction:

Pop

---

