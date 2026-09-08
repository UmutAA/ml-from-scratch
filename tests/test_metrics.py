import math
import numpy as np
import pytest

from metrics.metrics import (
    calculate_mse,
    calculate_rmse,
    sigmoid,
    calculate_bce,
    confusion_matrix,
    classification_report,
)

class TestMSE:
    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0]) # single error of 1 -> squared error 1
        # MSE = (0 + 0 + 1) / 3
        assert calculate_mse(y_true, y_pred) == pytest.approx(1 / 3)

    def test_perfect_predictions_give_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_mse(y, y) == 0.0

    def test_empty_input_raises_error(self):
        with pytest.raises(ZeroDivisionError):
            calculate_mse(np.array([]), np.array([]))

    def test_accept_python_lists(self):
        # y_trues / y_predicts aren't guaranteed to already be numpy arrays
        assert calculate_mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

class TestRMSE:
    def test_rmse_is_sqrt_of_mse(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([2, 2, 2, 2])
        # MSE = 4, RMSE = 2
        assert calculate_rmse(y_true, y_pred) == pytest.approx(2.0)

    def test_matches_manual_calculation(self):
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        expected = math.sqrt(calculate_mse(y_true, y_pred))
        assert calculate_rmse(y_true, y_pred) == pytest.approx(expected)

class TestSigmoid:
    def test_sigmoid_of_zero_is_one_half(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_output_is_bounded(self):
        z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        p = sigmoid(z)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)

    def test_sigmoid_is_monotonically_increasing(self):
        z = np.linspace(-10, 10, 50)
        p = sigmoid(z)
        assert np.all(np.diff(p) >= 0)

    def test_extreme_values_do_not_produce_nan_or_inf(self):
        # Large positive/negative z historically causes overflow in exp(-z)
        # if not handled/clipped correctly.
        z = np.array([-1e6, 1e6])
        p = sigmoid(z)
        assert not np.isnan(p).any()
        assert not np.isinf(p).any()

    class TestBCE:
        def test_confident_correct_predictions_give_low_loss(self):
            y = np.array([1.0, 0.0])
            p = np.array([0.999, 0.001])
            assert calculate_bce(y, p) < 0.01

        def test_confident_wrong_predictions_give_high_loss(self):
            y = np.array([1.0, 0.0])
            p = np.array([0.001, 0.999])
            assert calculate_bce(y, p) > 5.0

        def test_empty_input_raises(self):
            with pytest.raises(ZeroDivisionError):
                calculate_bce(np.array([]), np.array([]))

class TestConfusionMatrix:
    def test_perfect_predictions_are_all_on_diagonal(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)
        assert np.trace(cm) == 4  # everything correct
        assert cm.sum() == 4

    def test_known_confusion_matrix(self):
        # 2 true negatives, 1 false positive, 1 false negative, 2 true positives
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        cm = confusion_matrix(y_true, y_pred)

        # classes are label-encoded in sorted order: 0 -> row/col 0, 1 -> row/col 1
        assert cm[0, 0] == 2  # true negatives
        assert cm[0, 1] == 1  # false positives
        assert cm[1, 0] == 1  # false negatives
        assert cm[1, 1] == 2  # true positives


class TestClassificationReport:
    def test_output_dict_has_expected_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        report = classification_report(y_true, y_pred, output_dict=True)

        assert "accuracy" in report
        assert "0" in report
        assert "1" in report
        assert set(report["0"].keys()) == {"precision", "recall", "f1_score", "support"}

    def test_perfect_predictions_give_accuracy_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        report = classification_report(y_true, y_pred, output_dict=True)

        assert report["accuracy"]["f1_score"] == pytest.approx(1.0)

    def test_labels_length_mismatch_raises(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError):
            classification_report(y_true, y_pred, labels=["only_one_label"])
