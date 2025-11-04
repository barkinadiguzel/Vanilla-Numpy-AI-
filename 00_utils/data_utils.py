"""
data_utils.py

Helper functions for data preprocessing in neural networks:
- normalize: scale features
- shuffle_data: shuffle dataset
- create_mini_batches: split data into mini-batches
"""

import numpy as np

def normalize(X):
    """Normalize features to mean 0 and std 1"""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def shuffle_data(X, y):
    """Shuffle features and labels in unison"""
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def create_mini_batches(X, y, batch_size):
    """Split dataset into mini-batches"""
    X, y = shuffle_data(X, y)
    mini_batches = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        mini_batches.append((X_batch, y_batch))
    return mini_batches
