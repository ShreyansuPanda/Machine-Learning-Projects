# 🎵 Music Genre Classification using PCA + SVM

Welcome to the **Music Genre Classifier**!  
This project combines **Principal Component Analysis (PCA)** and **Support Vector Machine (SVM)** to classify songs into genres using audio-based features. This hybrid approach improves classification efficiency and accuracy, especially when dealing with high-dimensional musical data.

---

## 🎯 Problem Statement

With the explosion of music content across platforms, genre classification plays a vital role in music recommendation and search systems. This project implements a **Music Genre Classification System** that uses **PCA for dimensionality reduction** and **SVM for classification**, enhancing performance while retaining accuracy.

---
## DataSet:
GTZAN Dataset - Music Genre Classification: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

## 🚀 Concepts Used

- **PCA (Principal Component Analysis):**  
  A technique used to reduce the number of input features by transforming the data into a set of principal components while retaining the most important variance.

- **Feature Engineering:**  
  Involves extracting key audio features like tempo, spectral centroid, zero crossing rate, etc., which are useful for genre classification.

- **SVM (Support Vector Machine):**  
  A powerful supervised learning algorithm that finds an optimal boundary between classes after PCA reduces feature dimensionality.

---

## 🔍 Code Functionality

### 1. **Data Loading & Preprocessing**
- Uses audio features dataset (e.g., simulated GTZAN or MFCC-extracted data)
- Extracts relevant numerical features such as:
  - `tempo`, `spectral centroid`, `zero crossing rate`, etc.
- Standardizes the dataset for uniform scaling

### 2. **PCA for Dimensionality Reduction**
- PCA reduces the dataset to fewer components while preserving maximum variance
- Helps eliminate noise and redundancy before classification

### 3. **SVM Training**
- Dataset is split into:
  - **Training Set (80%)**
  - **Testing Set (20%)**
- Trained using `SVC(kernel='rbf')` or similar configuration

### 4. **Prediction and Evaluation**
- Predicts genre labels for test data
- Evaluation metrics include:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```
### 📌 Steps to Run
1. Clone the repository
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```
2. Navigate to the project directory
```bash
cd "Machine-Learning-Projects/Music Genre Classification using PCA + SVM"
```
3. Open CMD and run the script
```bash
python music_genre_classifier.py
```
---
## 🗂️ File Structure
```sh
Music Genre Classification using PCA + SVM/
│
├── Data                                # Data folder from the dataset
├── music_features.csv                  # CSV file with extracted audio features
├── music_genre_classifier.py           # Main Python script

```

---
## 🌟 Show Your Support
If you like this project, consider giving it a ⭐ on GitHub. Contributions are also welcome!
