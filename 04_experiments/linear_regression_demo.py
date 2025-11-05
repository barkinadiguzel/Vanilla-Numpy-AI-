"""
linear_regression_demo.py

Simple demo for Linear Regression using synthetic data.
Shows how the model learns a linear relationship y = ax + b.
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# ---- Generate sample data ----
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true relation: y = 3x + 4 + noise

# ---- Train model ----
model = LinearRegression(lr=0.01, epochs=2000)
model.fit(X, y)

# ---- Predict ----
y_pred = model.predict(X)

# ---- Plot results ----
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_pred, color="red", label="Fitted line")
plt.legend()
plt.title("Linear Regression Demo")
plt.show()
