"""
manual_weight_demo.py

A minimal demo showing how weights update step-by-step 
using plain gradient descent, no libraries or layers.
"""

import numpy as np

# simple dataset: y = 2x + 1 (with small noise)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(20, 1) * 0.5

# initialize parameters
w = np.random.randn(1, 1)
b = np.zeros((1, 1))
lr = 0.01
epochs = 200

for epoch in range(epochs):
    # forward pass
    y_pred = np.dot(X, w) + b

    # loss (mean squared error)
    loss = np.mean((y - y_pred) ** 2)

    # gradients
    dw = -2 * np.dot(X.T, (y - y_pred)) / len(X)
    db = -2 * np.mean(y - y_pred)

    # update weights
    w -= lr * dw
    b -= lr * db

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w[0][0]:.3f}, b: {b[0][0]:.3f}")

print(f"\nLearned parameters → w ≈ {w[0][0]:.2f}, b ≈ {b[0][0]:.2f}")
