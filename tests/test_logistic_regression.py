import numpy as np
import pytest

from models.logistic_regression import LogisticRegressionModel

class TestLogisticRegressionModel:
    def test_predictions_match_true_labels_on_separable_data(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionModel(learning_rate=0.5, epochs=2000)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.95

    def test_predict_returns_boolean_when_requested(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X, boolean=True)
        assert preds.dtype == bool

    def test_predict_returns_int_by_default(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.dtype in (np.int64, np.int32, int)

    def test_predict_shape_matches_input_rows(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_loss_decreases_over_training(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionModel(learning_rate=0.5, epochs=200)
        model.fit(X, y)

        assert model.loss_history[-1] < model.loss_history[0]

    def test_print_formula_before_fit_raises(self):
        model = LogisticRegressionModel()
        with pytest.raises(RuntimeError):
            model.print_formula()
