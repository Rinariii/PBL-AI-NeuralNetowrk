import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []
        self.mae_history = []
        self.r2_history = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred, lr):
        error = y_pred - y

        dW2 = self.a1.T @ error
        db2 = error.sum(axis=0, keepdims=True)

        dA1 = error @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.z1)

        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            y_pred = self.forward(X)

            loss = np.mean((y - y_pred) ** 2)
            mae = mean_absolute_error(y, y_pred)

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            self.loss_history.append(loss)
            self.mae_history.append(mae)
            self.r2_history.append(r2)

            self.backward(X, y, y_pred, lr)
