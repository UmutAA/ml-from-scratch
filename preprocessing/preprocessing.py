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