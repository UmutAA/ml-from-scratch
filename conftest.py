import pytest
import numpy as np

@pytest.fixture
def separable_data():
    # A clearly linearly separable dataset: x < 0 -> class 0, x > 0 -> class 1
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(200, 1))
    y = (X.flatten() > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_separable_data():
    np.random.seed(42)
    n_per_class = 50

    X0 = np.random.uniform(-3, -1.2, size=(n_per_class, 1))
    X1 = np.random.uniform(-0.8, 0.8, size=(n_per_class, 1))
    X2 = np.random.uniform(1.2, 3, size=(n_per_class, 1))

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    shuffle_idx = np.random.permutation(X.shape[0])
    return X[shuffle_idx], y[shuffle_idx]


@pytest.fixture
def multiclass_string_labels_data():
    np.random.seed(0)
    X = np.array([[-2.0], [-1.8], [-0.2], [0.1], [1.9], [2.2]])
    y = np.array(["cat", "cat", "dog", "dog", "bird", "bird"])
    return X, y

@pytest.fixture
def perfect_linear_data():
    # y = 3x + 5, no noise -> the model should recover these exactly
    np.random.seed(0)
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = 3 * X.flatten() + 5
    return X, y