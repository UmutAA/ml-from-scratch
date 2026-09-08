import numpy as np
import math
import pandas as pd
from preprocessing.preprocessing import LabelEncoder

def calculate_mse(y_trues: np.ndarray | list[float], y_predicts: np.ndarray | list[float]) -> float:
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

def calculate_rmse(y_trues: np.ndarray | list[float], y_predicts: np.ndarray | list[float]) -> float:
    return math.sqrt(calculate_mse(y_trues, y_predicts))

def sigmoid(z: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-z))
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return p

def calculate_bce(y: np.ndarray | list[float], p: np.ndarray | list[float]) -> float:
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

def confusion_matrix(y_trues: np.ndarray | list[float], y_predicts: np.ndarray | list[float]) -> np.ndarray:
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

def classification_report(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float], labels:list[str] | None = None, output_dict: bool = False) -> dict[str,dict[str,float]] | pd.DataFrame:
    """
    Calculates classification report and returns as either
    dict (output_dict = True) or pd.DataFrame (output_dict = False)
    """

    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)

    unique_classes = np.unique(np.concatenate((y_true, y_pred)))

    if labels is not None:
        if len(labels) != len(unique_classes):
            raise ValueError("labels' size must match the number of unique classes")
    else:
        labels = [str(c) for c in unique_classes]

    report = {}
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy: float

    if y_true.shape[0] > 0:
        accuracy = np.trace(conf_matrix) / float(y_true.shape[0])
    else:
        accuracy = 0.0

    for i, label in enumerate(labels):
        tp = conf_matrix[i, i]
        pred_sum = np.sum(conf_matrix[:, i])  # Column sum (TP + FP)
        true_sum = np.sum(conf_matrix[i, :])  # Row sum (TP + FN)

        precision = tp / pred_sum if pred_sum > 0 else 0.0
        recall = tp / true_sum if true_sum > 0 else 0.0

        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        report[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "support": int(true_sum)
        }

    report["accuracy"] = {
        "precision": None,
        "recall": None,
        "f1_score": float(accuracy),
        "support": int(y_true.shape[0])
    }

    if not output_dict:
        report = pd.DataFrame(report).T

    return report