import numpy as np
import pytest

from models.logistic_regression import (BinaryLogisticRegressionModel,
                                        MultinomialLogisticRegressionModel)

class TestBinaryLogisticRegressionModel:
    def test_predictions_match_true_labels_on_separable_data(self, separable_data):
        X, y = separable_data
        model = BinaryLogisticRegressionModel(learning_rate=0.5, epochs=2000)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.95

    def test_predict_returns_boolean_when_requested(self, separable_data):
        X, y = separable_data
        model = BinaryLogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X, boolean=True)
        assert preds.dtype == bool

    def test_predict_returns_int_by_default(self, separable_data):
        X, y = separable_data
        model = BinaryLogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.dtype in (np.int64, np.int32, int)

    def test_predict_shape_matches_input_rows(self, separable_data):
        X, y = separable_data
        model = BinaryLogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_loss_decreases_over_training(self, separable_data):
        X, y = separable_data
        model = BinaryLogisticRegressionModel(learning_rate=0.5, epochs=200)
        model.fit(X, y)

        assert model.loss_history[-1] < model.loss_history[0]

    def test_print_formula_before_fit_raises(self):
        model = BinaryLogisticRegressionModel()
        with pytest.raises(RuntimeError):
            model.print_formula()


class TestMultinomialLogisticRegressionModel:
    def test_predictions_match_true_labels_on_separable_data(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=3000)
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.90

    def test_predict_shape_matches_input_rows(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_returns_original_class_labels(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=500)
        model.fit(X, y)

        preds = model.predict(X)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    def test_predict_with_string_labels(self, multiclass_string_labels_data):
        X, y = multiclass_string_labels_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=2000)
        model.fit(X, y)

        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({"cat", "dog", "bird"})
        accuracy = np.mean(preds == y)
        assert accuracy == 1.0

    def test_weight_matrix_shape(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=200)
        model.fit(X, y)

        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        assert model.w.shape == (n_features, n_classes)
        assert model.b.shape == (n_classes,)

    def test_loss_decreases_over_training(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=200)
        model.fit(X, y)

        assert model.loss_history[-1] < model.loss_history[0]

    def test_loss_history_has_no_none_values(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=50)
        model.fit(X, y)

        assert all(loss is not None for loss in model.loss_history)
        assert all(not np.isnan(loss) for loss in model.loss_history)

    def test_print_formula_before_fit_raises(self):
        model = MultinomialLogisticRegressionModel()
        with pytest.raises(RuntimeError):
            model.print_formula()

    def test_print_formula_after_fit_does_not_raise(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=100)
        model.fit(X, y)

        model.print_formula()

    def test_converges_within_epoch_limit(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=10000, print_rate=10000)
        model.fit(X, y,tol=1e-6)

        assert len(model.loss_history) < 10000

    def test_multivariate_input_raises_on_plot(self, multiclass_separable_data):
        X, y = multiclass_separable_data
        X_multivariate = np.hstack([X, X])

        model = MultinomialLogisticRegressionModel(learning_rate=0.5, epochs=100)
        model.fit(X_multivariate, y)

        model.plot(X_multivariate, y)