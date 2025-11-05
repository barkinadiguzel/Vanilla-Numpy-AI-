"""
gradient_descent.py

Basic implementations of Gradient Descent (GD) and Stochastic Gradient Descent (SGD).
These functions update parameters (weights and biases) based on computed gradients.
"""

import numpy as np

def gradient_descent(params, grads, lr):
    """
    Vanilla Gradient Descent.
    params: dict of weights/biases
    grads: dict of gradients
    lr: learning rate
    """
    for key in params.keys():
        params[key] -= lr * grads[key]
    return params


def stochastic_gradient_descent(params, grads, lr):
    """
    Stochastic Gradient Descent (SGD).
    Same as GD but used per sample instead of full batch.
    """
    for key in params.keys():
        params[key] -= lr * grads[key]
    return params
