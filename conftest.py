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
def perfect_linear_data():
    # y = 3x + 5, no noise -> the model should recover these exactly
    np.random.seed(0)
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = 3 * X.flatten() + 5
    return X, y