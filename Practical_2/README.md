# 🧠 Word Embeddings using Word2Vec

## 📌 Overview

This practical demonstrates **Word Embeddings using Word2Vec** with two training approaches:

- **CBOW (Continuous Bag of Words)**
- **Skip-gram**

The practical uses **Gensim** to train word embedding models on a custom collection of sentences. The learned vector representations are then used to explore semantic relationships and similarities between words.

---

## 🎯 Objectives

- Implement Word2Vec using **CBOW**.
- Implement Word2Vec using **Skip-gram**.
- Generate vector representations of words.
- Calculate semantic similarity between words.
- Compare CBOW and Skip-gram similarity results.
- Find words with similar vector representations.
- Perform basic word analogy using vector relationships.
- Visualize word embeddings using **PCA**.
- Visualize relationships between words using a **similarity matrix**.

---

## 🔍 Practical Features

### 1. CBOW

The CBOW model learns to predict a target word from the surrounding context words.

### 2. Skip-gram

The Skip-gram model learns to predict surrounding context words from a target word.

### 3. Semantic Similarity

The trained models are used to calculate similarity between words such as:

- `queen` and `king`
- `apple` and `mango`
- `queen` and `apple`

### 4. Similar Word Analysis

The `most_similar()` function is used to identify words that have similar vector representations.

### 5. Word Analogy

The practical demonstrates vector-based word relationships using an analogy such as:

```text
king - man + woman

6. PCA Visualization

Word vectors are reduced to two dimensions using Principal Component Analysis (PCA) and plotted to visualize relationships between words.

7. Similarity Matrix

A similarity matrix is generated using word vectors and visualized with Plotly.

🛠️ Technologies Used
Python
Gensim
NumPy
Pandas
Plotly
Matplotlib
Scikit-learn
Jupyter Notebook / Google Colab
📚 Python Libraries
gensim
numpy
pandas
plotly
matplotlib
scikit-learn
📂 Project Structure
Practical_2/
│
├── GenAI_02.ipynb
└── README.md
⚙️ Installation

Install the required libraries using:

!pip install -q gensim plotly pandas matplotlib scikit-learn
🚀 Running the Practical

Open GenAI_02.ipynb in Jupyter Notebook or Google Colab.

Run the notebook cells in sequence to:

Install the required libraries.
Prepare the training sentences.
Train the CBOW model.
Train the Skip-gram model.
Calculate semantic similarities.
Find similar words.
Visualize word embeddings using PCA.
Perform word analogy.
Generate the word similarity matrix.
📊 Expected Output

The practical produces:

CBOW similarity scores
Skip-gram similarity scores
Similar-word results
PCA visualization of word embeddings
Word analogy results
Word similarity matrix
👩‍💻 Author

Anjali Patalbansi