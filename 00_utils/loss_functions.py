"""
loss_functions.py

This file contains common loss functions for neural networks.
Each function has forward computation and derivative for backpropagation.

- MSE: for regression tasks
- Cross-Entropy: for classification tasks
"""

import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    """Derivative of MSE w.r.t predictions"""
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    """Cross-Entropy Loss"""
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_true, y_pred):
    """Derivative of Cross-Entropy w.r.t predictions"""
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return - (y_true / y_pred) / y_true.shape[0]
