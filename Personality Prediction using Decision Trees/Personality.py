# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset (Big Five Personality Traits)
data = {
    "Openness": [7, 8, 6, 5, 9, 4, 6, 3, 8, 7, 6, 5, 4, 9, 7, 8, 5, 6, 7, 6],
    "Conscientiousness": [6, 5, 7, 8, 6, 9, 4, 6, 7, 5, 8, 6, 9, 4, 6, 7, 5, 8, 6, 9],
    "Extraversion": [5, 6, 8, 4, 7, 3, 5, 6, 7, 8, 4, 5, 7, 6, 8, 7, 6, 5, 8, 7],
    "Agreeableness": [8, 7, 6, 9, 5, 8, 7, 6, 5, 4, 9, 5, 8, 7, 6, 5, 7, 6, 5, 4],
    "Neuroticism": [3, 4, 6, 2, 5, 7, 6, 8, 3, 4, 6, 7, 5, 8, 6, 7, 5, 6, 4, 5],
    "Personality": ["Extrovert", "Extrovert", "Introvert", "Introvert", "Extrovert",
                    "Introvert", "Introvert", "Extrovert", "Extrovert", "Introvert",
                    "Introvert", "Extrovert", "Extrovert", "Introvert", "Extrovert",
                    "Extrovert", "Introvert", "Introvert", "Extrovert", "Introvert"]
}

# Convert dataset into DataFrame
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df.drop(columns=["Personality"])  # Independent Variables
y = df["Personality"]  # Target Variable

# Split data into Training and Testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
plt.show()
