# 🧠 Vanilla-Numpy-AI

A minimal neural network framework built **from scratch using only NumPy**.  
Includes basic models, layers, optimizers, and visualization helpers — all coded manually to demonstrate the inner workings of AI models.

---

## 📂 Project Structure
```
Vanilla-Numpy-AI/
│
├── 00_utils/ # Core utilities
│ ├── activation_functions.py → Sigmoid, ReLU, Tanh, Softmax
│ ├── loss_functions.py → MSE, Cross-Entropy
│ ├── data_utils.py → Normalization, batching, dummy datasets
│ └── init_utils.py → Weight & bias initialization
│
├── 01_models/ # Example models
│ ├── linear_regression.py
│ ├── logistic_regression.py
│ ├── simple_nn.py
│ └── simple_nn_manual.py
│
├── 02_layers/
│ └── dense_layer.py → Fully-connected layer implementation
│
├── 03_optimization/ # Optimizers
│ ├── gradient_descent.py → Vanilla GD, SGD
│ └── optimizers.py → Momentum, RMSProp, Adam
│
├── 04_experiments/ # Demo scripts
│ ├── linear_regression_demo.py
│ ├── simple_classification.py
│ └── manual_weight_demo.py
│
├── 05_visualizations/
│ └── plot_helpers.py → Decision boundaries, loss curves
│
├── requirements.txt # Only numpy
└── README.md

```
---




## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Vanilla-Numpy-AI.git
   cd Vanilla-Numpy-AI
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run a demo (for example):

   ```bash
   python 04_experiments/simple_classification.py
   ```
--- 

## 🧩 Features

- Fully connected neural networks built manually

- Backpropagation implemented from scratch

- Gradient Descent, Momentum, RMSProp, Adam

- Visualization utilities for loss & decision boundaries

- Educational design — ideal for learning fundamentals
---

### 🔹 Notes

This project is intentionally simple.
It’s meant for educational purposes, not for production use.
You can easily extend it by adding:

- Convolutional layers

- Dropout / BatchNorm

- More advanced optimizers

---
## 📬Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)


