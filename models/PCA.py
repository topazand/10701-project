import numpy as np

class PCA:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        
    def select(self, p):
        self.p = p
        self.components = self.eigenvectors[:, :self.p]

    def var_explained(self):
        return self.eigenvalues / np.sum(self.eigenvalues)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def inv_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean