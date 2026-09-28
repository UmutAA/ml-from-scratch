import numpy as np
import pytest

from models.linear_regression import LinearRegressionModel, Ridge, Lasso

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

def _ols(X, y):
    m = LinearRegressionModel(learning_rate=0.1, epochs=5000, print_rate=10**9)
    m.fit(X, y, tol=1e-14)
    return m

class TestLasso:
    def test_alpha_effect_on_coefficients(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1]

        model_small = Lasso(alpha=0.01)
        model_small.fit(X, y)

        model_large = Lasso(alpha=2.0)
        model_large.fit(X, y)

        assert np.sum(np.abs(model_large.w)) < np.sum(np.abs(model_small.w))

    def test_predictions_shape(self):
        X_train = np.random.randn(40, 4)
        y_train = np.random.randn(40)
        X_test = np.random.randn(10, 4)

        model = Lasso(alpha=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (10,) or predictions.shape == (10, 1)


class TestRidge:
    def test_alpha_effect_on_coefficients(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1]

        model_small = Ridge(alpha=0.001, learning_rate=0.01, epochs=1000)
        model_small.fit(X, y)

        model_large = Ridge(alpha=1.0, learning_rate=0.01, epochs=1000)
        model_large.fit(X, y)

        assert np.sum(np.abs(model_large.w)) < np.sum(np.abs(model_small.w))

    def test_predictions_shape(self):
        X_train = np.random.randn(40, 4)
        y_train = np.random.randn(40)
        X_test = np.random.randn(10, 4)

        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (10,) or predictions.shape == (10, 1)

    def test_coefficients_shrink_but_not_zero(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 5.0 * X[:, 0] + np.random.randn(50) * 0.1

        model = Ridge(alpha=5.0)
        model.fit(X, y)

        assert np.abs(model.w[1]) < 1.0
        assert model.w[1] != 0.0
