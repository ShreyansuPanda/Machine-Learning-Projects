# 📰 Fake News Detection using Perceptron

Welcome to the **Fake News Detector**!  
This project uses the **Perceptron algorithm**, a fundamental linear classifier, to detect fake news articles. With the help of **Natural Language Processing (NLP)** techniques and **text classification**, the model effectively categorizes news headlines or snippets as either *Fake* or *Real*.

---

## 🎯 Problem Statement

The rise of misinformation has created a demand for automatic tools to verify the authenticity of news content. This project builds a **Fake News Detection System** using the **Perceptron learning algorithm**, combining it with NLP techniques such as **TF-IDF vectorization** and basic text cleaning for accurate classification.

---

## 🚀 Concepts Used

- **Perceptron Algorithm:**  
  A linear classification algorithm that adjusts weights when it misclassifies an instance, forming the basis of neural networks.

- **Text Classification:**  
  Converts raw text into structured numerical representations to classify as "Fake" or "Real."

- **NLP (Natural Language Processing):**  
  Involves tokenization, stopword removal, and vectorization of text using techniques like TF-IDF.

---

## 🔍 Code Functionality

### 1. **Data Preparation**
- Loads a dataset with news text and labels (`fake`, `real`)
- Applies basic **NLP preprocessing**:
  - Lowercasing
  - Tokenization
  - Stopword removal
  - TF-IDF vectorization

### 2. **Perceptron Implementation**
- Either:
  - Implements a **custom perceptron**, or
  - Uses `Perceptron` from `scikit-learn`
- Trains using iterative weight updates based on classification error

### 3. **Classification & Evaluation**
- Data split into:
  - **Training Set (80%)**
  - **Testing Set (20%)**
- The model:
  - Trains on vectorized data
  - Predicts news authenticity
- Evaluation includes:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn nltk
```
### 📌 Steps to Run
1. Clone the repository
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```
2. Navigate to the project directory
```bash
cd "Machine-Learning-Projects/Fake News Detection using Perceptron"
```
3. Open CMD and run the script
```bash
python fake_news_detector.py
```
---
## 🗂️ File Structure
```sh
Fake News Detection using Perceptron/
│
├── fake_news.csv                      # Dataset with news text and labels
├── true_news.csv                      # Dataset with news text and labels
└── fake_news_detector.py              # Main Python script
```
--- 
## 🌟 Show Your Support
If you like this project, consider giving it a ⭐ on GitHub. Contributions are also welcome!
