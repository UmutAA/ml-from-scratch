import numpy as np

def calculate_mse(y_trues: np.ndarray, y_predicteds: np.ndarray) -> float:
    """
    Calculates and returns mean squared errors
    """

    mse_arr = np.array(y_trues - y_predicteds)
    mse_arr **= 2.0
    sum = 0.0
    for i in mse_arr:
        sum += i

    mse = sum / float(len(mse_arr))

    return mse
    