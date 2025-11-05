"""
dense_layer.py

Defines a simple Dense (fully connected) layer with forward and backward methods.
This layer is reusable — can be stacked to build larger networks.
"""

import numpy as np
from activation_functions import sigmoid, sigmoid_derivative

class Dense:
    def __init__(self, input_dim, output_dim, activation="sigmoid"):
        # Initialize weights and biases
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.activation = activation

    def forward(self, X):
        # Linear transformation
        self.input = X
        self.Z = np.dot(X, self.W) + self.b
        
        # Apply activation
        if self.activation == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation == "tanh":
            self.A = np.tanh(self.Z)
        elif self.activation == "relu":
            self.A = np.maximum(0, self.Z)
        else:
            self.A = self.Z  # linear
        return self.A

    def backward(self, dA, lr=0.1):
        # Compute derivative of activation
        if self.activation == "sigmoid":
            dZ = dA * sigmoid_derivative(self.Z)
        elif self.activation == "tanh":
            dZ = dA * (1 - np.tanh(self.Z)**2)
        elif self.activation == "relu":
            dZ = dA * np.where(self.Z > 0, 1, 0)
        else:
            dZ = dA
        
        # Gradients
        m = self.input.shape[0]
        self.dW = np.dot(self.input.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Gradient for previous layer
        dA_prev = np.dot(dZ, self.W.T)
        
        # Update parameters
        self.W -= lr * self.dW
        self.b -= lr * self.db
        
        return dA_prev
