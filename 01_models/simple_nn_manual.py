"""
simple_nn_manual.py

A simple **2-layer neural network** built completely from scratch (manual backprop).
Dataset: XOR problem (outputs 1 when inputs differ, 0 when they’re the same).
Goal: Show step-by-step how forward pass, loss computation, backpropagation,
and parameter updates work without using any ML libraries.

Usage: Run this file to train the network and see XOR predictions.
"""

import numpy as np

# XOR data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialization
np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.01   # weights for layer 1
b1 = np.zeros((1, 3))               # bias for layer 1
W2 = np.random.randn(3, 1) * 0.01   # weights for layer 2
b2 = np.zeros((1, 1))               # bias for layer 2

lr = 0.1         # learning rate
epochs = 10000   # training iterations

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training loop
for epoch in range(epochs):
    # ---- Forward pass ----
    Z1 = np.dot(X, W1) + b1        # linear step for hidden layer
    A1 = sigmoid(Z1)               # activation for hidden layer
    Z2 = np.dot(A1, W2) + b2       # linear step for output layer
    A2 = sigmoid(Z2)               # activation for output layer
    
    # ---- Compute loss ----
    loss = mse(y, A2)
    
    # ---- Backward pass ----
    dA2 = -(y - A2)                                 # gradient of loss wrt output
    dZ2 = dA2 * sigmoid_derivative(Z2)              # gradient at output layer
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]            # gradient wrt W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]  # gradient wrt b2
    
    dA1 = np.dot(dZ2, W2.T)                         # backprop into hidden layer
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
    
    # ---- Update weights ----
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    # ---- Show progress ----
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---- Final predictions ----
predictions = (A2 >= 0.5).astype(int)
print("\nPredictions:")
print(predictions)
print("\nTrue labels:")
print(y)
