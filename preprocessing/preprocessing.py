import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X: np.ndarray):
        """
        Learns mean and standard deviation in data
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scales the data with fitted mean and standard deviation values
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scales has not been fitted yet! Must call .fit() method first")
        
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean) / self.scale)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Learns the mean and standard deviation and scales the data
        """
        return self.fit(X).transform(X)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into 2 groups as train and test
    Returns X_train, X_test, y_train, y_test
    """
    sample_size = X.shape[0]
    indices = np.random.permutation(sample_size)
    test_count = int(sample_size * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]