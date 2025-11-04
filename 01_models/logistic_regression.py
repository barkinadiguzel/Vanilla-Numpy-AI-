"""
logistic_regression.py

Simple Logistic Regression using numpy.
Supports forward pass with sigmoid, binary cross-entropy loss,
and gradient descent optimization.
"""

import numpy as np
from data_utils import create_mini_batches
from loss_functions import cross_entropy, cross_entropy_derivative
from activation_functions import sigmoid
from init_utils import random_init

class LogisticRegression:
    def __init__(self, input_dim, lr=0.01):
        self.W = random_init((input_dim, 1))
        self.b = 0.0
        self.lr = lr

    def forward(self, X):
        """Forward pass: predict probabilities"""
        z = np.dot(X, self.W) + self.b
        return sigmoid(z)

    def backward(self, X, y_true, y_pred):
        """Compute gradients and update weights"""
        m = X.shape[0]
        dZ = cross_entropy_derivative(y_true, y_pred)  # derivative of loss
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ) / m

        # update weights
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def train(self, X, y, epochs=1000, batch_size=None, verbose=False):
        """Train model with gradient descent"""
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
                loss = cross_entropy(y, y_pred_full)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Return predicted probabilities"""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Return binary predictions"""
        probs = self.forward(X)
        return (probs >= threshold).astype(int)
