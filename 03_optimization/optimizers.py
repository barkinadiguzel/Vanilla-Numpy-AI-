"""
optimizers.py

Implements common optimization algorithms: Momentum, RMSProp, and Adam.
Each one updates weights more efficiently than plain Gradient Descent
by smoothing or scaling gradients over time.

This part may look complex, but understanding the main idea — 
how they adjust learning dynamically — is enough for now.
You can dive deeper into the math later if you want.
"""
import numpy as np


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.v:
                self.v[key] = np.zeros_like(grads[key])
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * grads[key]
            params[key] -= self.lr * self.v[key]
        return params


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.s:
                self.s[key] = np.zeros_like(grads[key])
            self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (grads[key] ** 2)
            params[key] -= self.lr * grads[key] / (np.sqrt(self.s[key]) + self.eps)
        return params


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])

            # Moving averages
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params
