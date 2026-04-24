import numpy as np

# to be implemented

class KFolder:
    # create k fold views for data and labels
    def __init__(self, data, labels, k=5):
        self.data = data
        self.labels = labels
        self.k = k

    def get_fold(self, fold_index):
        fold_size = len(self.data) // self.k
        start = fold_index * fold_size
        end = start + fold_size
        X_train = np.concatenate((self.data[:start, :], self.data[end:, :]), axis=0)
        y_train = np.concatenate((self.labels[:start], self.labels[end:]), axis=0)
        X_val = self.data[start:end, :]
        y_val = self.labels[start:end]
        return X_train, y_train, X_val, y_val
    
    def split(self):
        for fold_index in range(self.k):
            yield self.get_fold(fold_index)