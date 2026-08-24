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