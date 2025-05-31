import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Ensure 'house_prices.csv' contains 'square_feet' and 'price' columns)
data = pd.read_csv("house_prices.csv")  
X = data['square_feet'].values
y = data['price'].values

# Feature Scaling (Standardization)
X = (X - X.mean()) / X.std()

# Add bias term (column of 1s)
X = np.c_[np.ones(X.shape[0]), X]

# Gaussian Kernel Function
def kernel(x, x_point, tau):
    return np.exp(-np.square(x - x_point) / (2 * tau**2))

# Locally Weighted Regression Function
def locally_weighted_regression(x_test, X, y, tau):
    m = X.shape[0]
    y_pred = np.zeros(len(x_test))

    for i, x_point in enumerate(x_test):
        weights = np.diag(kernel(X[:, 1], x_point, tau))  # Compute weight matrix
        theta = np.linalg.pinv(X.T @ weights @ X) @ X.T @ weights @ y  # Compute parameters
        y_pred[i] = np.dot([1, x_point], theta)  # Predict price

    return y_pred

# Generate test points for prediction
x_test = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)

# Predict house prices using LWR (tau = 0.5)
tau = 0.5
y_pred = locally_weighted_regression(x_test, X, y, tau)

# Visualization
plt.scatter(X[:, 1], y, label="Training Data")
plt.plot(x_test, y_pred, color='red', label="LWR Prediction")
plt.xlabel("Square Feet (Normalized)")
plt.ylabel("Price")
plt.legend()
plt.title("House Price Prediction using Locally Weighted Regression")
plt.show()
