"""
activation_functions.py

This file contains common activation functions used in neural networks.
Each function includes its forward computation and derivative for backpropagation.

- Sigmoid: squashes input to range [0,1], useful for probabilities.
- Tanh: squashes input to range [-1,1], centered around 0.
- ReLU: returns max(0, x), simple and effective for hidden layers.
- Softmax: converts a vector of numbers into probabilities that sum to 1.
"""

import numpy as np

def sigmoid(x):
    """Sigmoid activation: output in [0,1]"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid for backprop"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation: output in [-1,1], zero-centered"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh for backprop"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU activation: output is 0 if x<0 else x"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU for backprop"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation: outputs probabilities summing to 1"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
    return e_x / e_x.sum(axis=1, keepdims=True)
