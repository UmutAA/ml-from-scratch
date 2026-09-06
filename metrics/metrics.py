import numpy as np
import math
import pandas as pd
from exceptions import IncompatibleDimension
from preprocessing.preprocessing import LabelEncoder


def calculate_mse(y_trues: np.ndarray, y_predicts: np.ndarray) -> float:
    """
    Calculates and returns mean squared errors
    """

    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)
    
    if not isinstance(y_predicts, np.ndarray):
        y_predicts = np.asarray(y_predicts)

    m = y_trues.shape[0]
    if m == 0:
        raise ZeroDivisionError("Dimension can't be 0")
        return

    mse_matrix = (y_trues - y_predicts) ** 2
    mse = np.sum(mse_matrix) / float(m)

    return float(mse)

def calculate_rmse(y_trues: np.ndarray, y_predicts: np.ndarray) -> float:
    return math.sqrt(calculate_mse(y_trues, y_predicts))

def sigmoid(z: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-z))
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return p

def calculate_bce(y: np.ndarray, p: np.ndarray) -> float:
    """
    Calculates and returns binary cross entropy error (Log loss)
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float64)
        
    if not isinstance(p, np.ndarray):
        p = np.asarray(p, dtype=np.float64)

    n = p.shape[0]

    if n == 0:
        raise ZeroDivisionError("Dimension can't be 0")
        return
    
    bce_matrix = y * np.log(p) + (1 - y) * np.log(1 - p)
    bce = - np.sum(bce_matrix) / float(n)

    return float(bce)

def confusion_matrix(y_trues: np.ndarray, y_predicts: np.ndarray) -> np.ndarray:
    """
    Calculates and returns confusion matrix
    """

    if not isinstance(y_trues, np.ndarray):
        y_trues = np.asarray(y_trues)

    if not isinstance(y_predicts, np.ndarray):
        y_predicts = np.asarray(y_predicts)

    uniques = np.unique(np.concatenate((y_trues, y_predicts)))

    num_classes = len(uniques)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    le = LabelEncoder()
    le.fit(uniques)

    y_trues_encoded = le.transform(y_trues)
    y_predicts_encoded = le.transform(y_predicts)

    for i in range (y_trues.shape[0]):
        conf_matrix[y_trues_encoded[i], y_predicts_encoded[i]] += 1

    return conf_matrix
