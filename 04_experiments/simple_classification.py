"""
simple_classification.py

Demo of a simple binary classification using Logistic Regression.
Shows how the model separates two classes in 2D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# ---- Generate synthetic data ----
np.random.seed(42)
num_samples = 100

# Class 0 (blue)
X0 = np.random.randn(num_samples, 2) + np.array([-2, -2])
# Class 1 (red)
X1 = np.random.randn(num_samples, 2) + np.array([2, 2])

# Combine
X = np.vstack((X0, X1))
y = np.array([0] * num_samples + [1] * num_samples).reshape(-1, 1)

# ---- Train model ----
model = LogisticRegression(lr=0.1, epochs=1000)
model.fit(X, y)

# ---- Predictions ----
y_pred = model.predict(X)

# ---- Plot results ----
plt.scatter(X[:, 0], X[:, 1], c=y_pred.flatten(), cmap="bwr", alpha=0.7)
plt.title("Simple Binary Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
