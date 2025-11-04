"""
linear_regression.py

Simple Linear Regression model using numpy.
Supports forward pass, mean squared error loss, and gradient descent optimization.
"""

import numpy as np
from data_utils import create_mini_batches
from loss_functions import mse, mse_derivative
from init_utils import random_init

class LinearRegression:
    def __init__(self, input_dim, lr=0.01):
        self.W = random_init((input_dim, 1))
        self.b = 0.0
        self.lr = lr

    def forward(self, X):
        """Forward pass: predict output"""
        return np.dot(X, self.W) + self.b

    def backward(self, X, y_true, y_pred):
        """Compute gradients and update weights using gradient descent"""
        m = X.shape[0]
        dW = np.dot(X.T, mse_derivative(y_true, y_pred)) / m
        db = np.sum(mse_derivative(y_true, y_pred)) / m

        # update weights
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def train(self, X, y, epochs=1000, batch_size=None, verbose=False):
        """Train the model using gradient descent"""
        for epoch in range(epochs):
            if batch_size:
                batches = create_mini_batches(X, y, batch_size)
            else:
                batches = [(X, y)]

            for X_batch, y_batch in batches:
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            if verbose and epoch % 100 == 0:
                y_pred_full = self.forward(X)
                loss = mse(y, y_pred_full)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """Predict new data"""
        return self.forward(X)
