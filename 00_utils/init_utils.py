"""
init_utils.py

Helper functions for initializing weights and biases in neural networks.
- random_init: small random values
- zeros_init: zeros
- xavier_init: Xavier/Glorot initialization for better convergence
"""

import numpy as np

def random_init(shape, scale=0.01):
    """Initialize weights with small random numbers"""
    return np.random.randn(*shape) * scale

def zeros_init(shape):
    """Initialize weights or biases with zeros"""
    return np.zeros(shape)

def xavier_init(shape):
    """
    Xavier/Glorot initialization
    Good for tanh/sigmoid activations
    """
    in_dim, out_dim = shape
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=shape)
