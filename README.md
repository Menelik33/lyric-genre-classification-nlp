# NLP Lyric Identifier: Genre Classification from Song Lyrics

***Predicting the music genre of a song based on its lyrics*** by implementing TensorFlow/Keras to develop a NLP pipeline that uses: neural text vectorization, neural network classification and word embedding techniques. 

This project demonstrates how unstructured textual data (in this case: song lyrics) can be converted into numerical feature representations in order to train a neural model capable of classifying the correct music genres based on the songs lyrics

---

## Project Overview

Song lyrics contain stylistic patterns (in particular: words used [text vectorization] and their definition [word embedding]) that often correlate with musical genres.  
This project builds a machine learning system that learns those patterns and predicts a song's genre from its lyrics.

The pipeline performs:

• Text preprocessing  
• Vocabulary construction  
• Text vectorization
word embedding
• Neural network classification  
• Model evaluation on validation and test datasets  

Example input:

"I'm standing on the corner with my guitar tonight"

Predicted output:

Country

---

## Dataset

The datasets contains song lyrics labeled with their corresponding genre.

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

The dataset is split into:

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

## 1. Target Encoding

The genre labels are converted into **one-hot encoded vectors**.

| Genre | Encoded Vector |
|------|------|
| Country | [1,0,0] |
| Rock | [0,1,0] |
| Hip-Hop | [0,0,1] |

This allows the neural network to perform **multi-class classification using softmax**.

---

## 2. Text Vectorization

Lyrics are converted into numerical feature vectors using:

keras.layers.TextVectorization

The vectorization layer:

• Scans the training corpus  
• Builds a vocabulary of the **5,000 most frequent words**  
• Assigns each word a numeric index  
• Maps unknown words to an **UNK token**

Example lyric:

"I love my guitar tonight"

Becomes a **multi-hot vector** where each vocabulary word present in the lyric is marked with a `1`.

Example representation (simplified):

[0,0,1,0,1,0,0,1,...]

This creates a **5,000-dimensional sparse feature vector** for each lyric.

---

## 3. Multi-Hot Text Representation

Instead of using sequences, the model uses **multi-hot encoding**, which represents each lyric by the presence of words in the vocabulary.

Advantages:

• Efficient representation of text features  
• Works well with feedforward networks  
• Captures word presence without requiring sequence modeling  

---

## 4. Genre Classification Model

The classifier is a **feedforward neural network** built using TensorFlow/Keras.

Architecture:

Input: multi-hot lyric vector (5000 features)  
↓  
Dense Layer  
↓  
Dropout Regularization  
↓  
Softmax Output Layer  

The output layer produces probabilities for each genre class.

Example output:

[0.02, 0.91, 0.07]

Which corresponds to:

Rock

---

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

"Got my boots and my truck tonight"

Model prediction:

Country

---

## Future Improvements

Potential extensions include:

• Word embeddings (Word2Vec / GloVe)  
• Transformer-based models (BERT)  
• Sequence models (LSTM / GRU)  
• Attention mechanisms for lyric interpretation  
• Larger vocabularies and dataset expansion  
