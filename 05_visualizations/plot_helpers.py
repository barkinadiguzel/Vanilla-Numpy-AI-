"""
plot_helpers.py

Utility functions for visualizing data, decision boundaries, and model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt


# ===== Plot 2D Data =====
def plot_data(X, y):
    """Visualize 2D classification data (two classes)."""
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='bwr', edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Data Visualization")
    plt.show()


# ===== Plot Decision Boundary =====
def plot_decision_boundary(model, X, y, resolution=0.02):
    """Visualize decision boundary for a trained model."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model(grid)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='bwr', edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


# ===== Plot Training Loss =====
def plot_loss(losses):
    """Plot loss curve over training epochs."""
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()
