# 🏠 House Price Prediction using Locally Weighted Regression (LWR)

Welcome to the **House Price Prediction Model**!  
This project uses **Locally Weighted Regression (LWR)** to predict house prices based on square footage, adapting to local patterns in the dataset for higher accuracy than standard linear regression.

---

## 🧠 Problem Statement

The price of real estate is influenced by numerous **local factors** such as location, property size, and nearby amenities. Traditional linear regression methods often fail to capture these nuances.  
This project builds a **flexible, instance-based prediction model** using **Locally Weighted Regression**, which tailors itself to each query point using nearby data for precise predictions.

---

## 🚀 Concepts Used

- **Regression:**  
  Models the relationship between house price (dependent variable) and square footage (independent variable).

- **Locally Weighted Regression (LWR):**  
  A non-parametric method that uses a **Gaussian kernel** to weigh data points by their distance from the query point and fits a local linear model.

- **Weighted Models:**  
  Gives more importance to data points near the query using a **weight matrix**, improving prediction accuracy.

- **Data Visualization:**  
  Plots both actual and predicted values to visually assess model performance and fit.

---

## 🔍 Code Functionality

### 1. **Data Loading & Normalization**
- Dataset: `house_prices.csv` containing:
  - `square_feet`: Size of the house
  - `price`: Corresponding house price
- The square footage data is **normalized** to avoid feature scaling issues during regression.

### 2. **Bias Term Addition**
- A column of ones is added to the feature matrix to include the **intercept** in the model equations.

### 3. **Gaussian Kernel Function**
- Computes weights for each training point based on **distance** from a test point using:
  - `tau`: Bandwidth parameter that controls the width of the kernel

### 4. **LWR Function**
- For each test point:
  - Builds a **diagonal weight matrix**
  - Solves the **weighted normal equation** for theta
  - Predicts the price using the learned theta

### 5. **Visualization**
- Displays a **scatter plot** of actual data and the **regression curve**, providing insight into the model’s performance.

---

## 💻 Run the Project

### 📥 Prerequisites

- Python 3.x
- Jupyter Notebook or any Python IDE
- Required packages:

```bash
pip install numpy pandas matplotlib
```

### 📌 Steps to Run
1. Clone the repository:
```bash
git clone https://github.com/ShreyansuPanda/Machine-Learning-Projects.git
```

2. Navigate to the project directory:
```bash
cd "Machine-Learning-Projects/House Price Prediction using LWR"
```
3. 3. Open CMD and run Python script
```bash
python House Price Prediction using Locally Weighted Regression.py
```
---

## 🗂️ File Structure
```sh
House Price Prediction using LWR/
│
├── house_prices.csv                # Dataset with square footage and price
├── house_price_prediction_lwr.py   # Python script for LWR
```
---

## 🌟 Show Your Support
If you like this project, consider giving it a ⭐ on GitHub. Contributions are also welcome!
