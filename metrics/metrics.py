import numpy as np
import math

def calculate_mse(y_trues: np.ndarray, y_predicteds: np.ndarray) -> float:
    """
    Calculates and returns mean squared errors
    """

    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)
    
    if not isinstance(y_predicteds, np.ndarray):
        y_predicteds = np.asarray(y_predicteds)

    m = y_trues.shape[0]
    if m == 0:
        raise ZeroDivisionError("Dimension can't be 0")

    mse_matrix = (y_trues - y_predicteds) ** 2
    mse = np.sum(mse_matrix) / float(m)

    return float(mse)

def calculate_rmse(y_trues: np.ndarray, y_predicteds: np.ndarray) -> float:
    return math.sqrt(calculate_mse(y_trues, y_predicteds))

def sigmoid(x: np.ndarray) -> float:
    z = 1 / (1 + np.exp(-x))
    return z

def calculate_bce(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates and returns binary cross entropy error (Log loss)
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float64)

    n = y.shape[0]

    if n == 0:
        raise ZeroDivisionError("Dimension can't be 0")

    p = sigmoid(x)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    
    bce_matrix = y * np.log(p) + (1 - y) * np.log(1 - p)
    bce = - np.sum(bce_matrix) / float(n)

    return float(bce)