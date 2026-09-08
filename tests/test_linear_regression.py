import numpy as np
import pytest

from models.linear_regression import LinearRegressionModel

class TestLinearRegressionModel:
    def test_learns_correct_slope_and_intercept(self, perfect_linear_data):
        X, y = perfect_linear_data
        model = LinearRegressionModel(learning_rate=0.01, epochs=5000)
        model.fit(X, y)

        assert model.w[0] == pytest.approx(3.0, abs=0.05)
        assert model.b == pytest.approx(5.0, abs=0.05)

    def test_predict_shape_matches_input_rows(self, perfect_linear_data):
        X, y = perfect_linear_data
        model = LinearRegressionModel(learning_rate=0.01, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_accepts_1d_array(self, perfect_linear_data):
        X, y = perfect_linear_data
        model = LinearRegressionModel(learning_rate=0.01, epochs=500)
        model.fit(X, y)

        preds = model.predict(np.array([1.0, 2.0, 3.0]))
        assert preds.shape == (3,)

    def test_loss_decreases_over_training(self, perfect_linear_data):
        X, y = perfect_linear_data
        model = LinearRegressionModel(learning_rate=0.01, epochs=200)
        model.fit(X, y)

        # Loss at the end should be (much) lower than at the start.
        assert model.loss_history[-1] < model.loss_history[0]

    def test_print_formula_before_fit_raises(self):
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError):
            model.print_formula()

    def test_accepts_multiple_features(self):
        # y = 2*x1 - 1*x2 + 4
        np.random.seed(1)
        X = np.random.uniform(-5, 5, size=(200, 2))
        y = 2 * X[:, 0] - 1 * X[:, 1] + 4

        model = LinearRegressionModel(learning_rate=0.01, epochs=3000)
        model.fit(X, y)

        assert model.w[0] == pytest.approx(2.0, abs=0.1)
        assert model.w[1] == pytest.approx(-1.0, abs=0.1)
        assert model.b == pytest.approx(4.0, abs=0.1)
