# ❤️ Heart Disease Prediction using Feature Engineering

Welcome to the **Heart Disease Predictor**!  
This project leverages **feature engineering** techniques to enhance the performance of classification algorithms in predicting heart disease. By selecting and transforming the most relevant features, the model provides better accuracy and interpretability for early detection of heart disease.

---

## 🎯 Problem Statement

Heart disease is a major cause of mortality worldwide. Early diagnosis can save lives by enabling timely treatment. This project builds a heart disease prediction system using **classification algorithms** and improved **feature engineering**, including **data preprocessing**, **feature scaling**, and **feature selection** to boost model accuracy.

---

## 🚀 Concepts Used

- **Feature Engineering:**  
  The process of cleaning, transforming, and selecting features to improve machine learning model performance.

- **Classification:**  
  Predicts whether a patient is likely to have heart disease using clinical indicators like age, cholesterol, and blood pressure.

- **Data Preprocessing:**  
  Cleans raw data by imputing missing values, encoding categorical variables, and scaling numerical values.

---

## 🔍 Code Functionality

### 1. **Data Loading & Exploration**
- Loads the **Heart Disease dataset**
- Visualizes distributions and correlation between features
- Target variable: presence (`1`) or absence (`0`) of heart disease

### 2. **Feature Preprocessing**
- **Handling Missing Values**:  
  Fills or removes null entries to maintain data integrity
- **Standardization / Normalization**:  
  Scales numerical values for model compatibility
- **Encoding**:  
  Converts categorical data (if any) into numerical format

### 3. **Feature Selection**
- Uses:
  - **Correlation Analysis**
  - **Tree-based feature importance (e.g., RandomForest)**
- Retains only the most informative features for model training

### 4. **Model Training**
- Uses classifiers such as:
  - **Logistic Regression**
  - **Decision Tree**
  - **SVM**
- Trains model using selected features on the training dataset

### 5. **Model Evaluation**
- Evaluates model performance using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Analyzes feature contributions to interpret predictions

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
### 📌 Steps to Run
1. Clone the repository
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```
2. Navigate to the project folder
```
cd "Machine-Learning-Projects/Heart Disease Prediction using Feature Engineering"
```
3. Run the script
```bash
python heart_disease_predictor.py
```
---
## 🗂️ File Structure
```sh
Heart Disease Prediction using Feature Engineering/
│
├── heart.csv                            # Dataset
├── heart_disease_predictor.py           # Python script
```
---
## 🌟 Show Your Support
If you found this project helpful, leave a ⭐ on GitHub and share your feedback! Contributions are welcome.
