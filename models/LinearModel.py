import numpy as np
import tqdm

import numpy as np
import tqdm

class LinearModel:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prob(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        p = self.prob(X)
        return (p >= 0.5).astype(int)

    def compute_loss(self, X, y):
        p = self.prob(X)
        loss = -np.mean(y * np.log(p + 1e-14) + (1 - y) * np.log(1 - p + 1e-14))
        return loss

    def fit(self, X, y, learning_rate=0.5, num_iterations=10000):
        m = len(y)
        progress_bar = tqdm.tqdm(range(num_iterations), desc="Training Linear Model")
        for i in progress_bar:
            p = self.prob(X)
            dw = (1/m) * np.dot(X.T, (p - y))
            db = (1/m) * np.sum(p - y)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            if i % 100 == 0:
                loss = self.compute_loss(X, y)
                progress_bar.set_description(f"Training Linear Model, loss:{loss:.4f}")
            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

        final_loss = self.compute_loss(X, y)
        print(f"Final Loss: {final_loss:.4f}")

        