# 🧠 Personality Prediction using Decision Trees

Welcome to the **Personality Prediction System**!  
This project applies **Decision Tree Classification** to predict personality types—**Introvert** or **Extrovert**—based on scores across key behavioral traits. The model provides interpretability through decision paths and feature importance.

---

## 🧠 Problem Statement

Understanding personality can enhance personalization in fields like **education**, **recruitment**, and **mental health**.  
This project builds a **Decision Tree-based classification model** that predicts personality type by analyzing behavioral scores based on the **Big Five traits**. The model delivers insights using interpretable decision boundaries and trait importance.

---

## 🚀 Concepts Used

- **Decision Trees:**  
  A supervised learning model that recursively splits the data to classify labels based on information gain or Gini index.

- **Classification:**  
  Predicts categorical labels—**Introvert** or **Extrovert**—using numerical trait scores.

- **Feature Importance:**  
  Identifies which traits (e.g., Extraversion, Openness) influence predictions the most.

---

## 🔍 Code Functionality

### 1. **Dataset Creation**
- A sample dataset is created with the **Big Five personality traits**:
  - `Openness`, `Conscientiousness`, `Extraversion`, `Agreeableness`, `Neuroticism`
- Each entry is labeled with a target class: `Introvert` or `Extrovert`

### 2. **Data Splitting**
- Dataset is split into:
  - **Training Set (80%)**
  - **Testing Set (20%)**  
  Using `train_test_split` for robust evaluation.

### 3. **Model Training**
- A `DecisionTreeClassifier` is initialized with:
  - `criterion='entropy'` to use **information gain**
  - `max_depth=3` to avoid overfitting
- Model is trained on the training dataset

### 4. **Prediction and Evaluation**
- Predictions are made on the test set
- Performance is evaluated using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-score)

### 5. **Visualization**
- The trained decision tree is visualized using:
  - `sklearn.tree.plot_tree`
- The plot reveals:
  - Which traits lead to specific personality predictions
  - How decisions are made at each split

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```
### 📌 Steps to Run
1. Clone the repository:
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```
2.Navigate to the project folder:
```bash
cd "Machine-Learning-Projects/Personality Prediction using Decision Trees"
```
3. Run the python script
```bash
python Personality.py
```
---
## 🌟 Show Your Support
If you enjoyed this project, don’t forget to ⭐ the repository and share your suggestions or enhancements!

 
