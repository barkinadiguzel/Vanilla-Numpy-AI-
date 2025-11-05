"""
simple_nn.py

A minimal 2-layer neural network using NumPy and class-based structure.
It learns the XOR function — outputs 1 if inputs differ, 0 if same.
This version is cleaner and modular compared to the manual one.
"""

import numpy as np

# Activation and loss
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Neural Network class
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dA2 = -(y - self.A2)
        dZ2 = dA2 * sigmoid_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mse(y, output)
            self.backward(X, y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

# ---- Dataset (XOR) ----
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# ---- Train model ----
nn = SimpleNN(input_dim=2, hidden_dim=3, output_dim=1, lr=0.1)
nn.train(X, y)

# ---- Predictions ----
print("\nPredictions:")
print(nn.predict(X))
print("\nTrue labels:")
print(y)
