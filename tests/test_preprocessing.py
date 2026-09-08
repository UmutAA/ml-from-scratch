import numpy as np
import pytest

from preprocessing.preprocessing import (
StandardScaler,
LabelEncoder,
train_test_split,
)

# ------------------- StandardScaler -------------------
class TestStandardScaler:
    def test_fit_transform_gives_zero_mean_unit_std(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-8)

    def test_transform_uses_stats_learned_from_fit(self):
        # transform() on new data must reuse the mean/std learned during fit(),
        # not recompute them from the new data.
        X_train = np.array([[0.0], [10.0]])
        X_new = np.array([[5.0]])

        scaler = StandardScaler()
        scaler.fit(X_train)
        result = scaler.transform(X_new)

        expected = (5.0 - scaler.mean) / scaler.scale
        assert np.allclose(result, expected)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([[1.0]]))

    def test_constant_column_does_not_divide_by_zero(self):
        # A column with zero variance would normally cause a divide-by-zero.
        # The implementation should guard against this (scale -> 1.0).
        X = np.array([[5.0], [5.0], [5.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()

    def test_accepts_1d_input(self):
        X = np.array([1.0, 2.0, 3.0])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == (3, 1)

# ------------------- TrainTestSplit -------------------

class TestTrainTestSplit:
    def test_split_sizes(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        assert X_test.shape[0] == 20
        assert X_train.shape[0] == 80
        assert y_test.shape[0] == 20
        assert y_train.shape[0] == 80

    def test_no_overlap_between_train_and_test(self):
        X = np.arange(50).reshape(-1, 1)
        y = np.arange(50)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        train_values = set(X_train.flatten().tolist())
        test_values = set(X_test.flatten().tolist())
        assert train_values.isdisjoint(test_values)

    def test_all_rows_are_preserved(self):
        X = np.arange(20).reshape(-1, 1)
        y = np.arange(20)

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25)
        combined = sorted(X_train.flatten().tolist() + X_test.flatten().tolist())
        assert combined == list(range(20))

    def test_X_and_y_stay_aligned(self):
        # y[i] must still correspond to the same original row as X[i] after splitting.
        X = np.arange(30).reshape(-1, 1)
        y = np.arange(30) * 10  # y is just X * 10, so we can check the relationship holds

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        assert np.array_equal(X_train.flatten() * 10, y_train)
        assert np.array_equal(X_test.flatten() * 10, y_test)

# ------------------- LabelEncoder -------------------

class TestLabelEncoder:
    def test_binary_labels_roundtrip(self):
        y = np.array(["cat", "dog", "cat", "dog"])
        le = LabelEncoder()
        encoded = le.fit_transform(y)

        assert set(encoded.tolist()) == {0, 1}
        assert le.n_classes_ == 2

    def test_multiclass_labels(self):
        # Regression test: fitting on more than 2 distinct classes used to raise
        y = np.array(["cat", "dog", "bird", "cat", "bird", "dog"])
        le = LabelEncoder()
        encoded = le.fit_transform(y)

        assert le.n_classes_ == 3
        assert set(encoded.tolist()) == {0, 1, 2}

    def test_unseen_label_raises_value_error(self):
        le = LabelEncoder()
        le.fit(np.array(["cat", "dog"]))

        with pytest.raises(ValueError):
            le.transform(np.array(["bird"]))

    def test_encoding_is_consistent(self):
        y = np.array(["b", "a", "c"])
        le = LabelEncoder()
        le.fit(y)

        # classes_ should be sorted, so a=0, b=1, c=2
        assert list(le.classes_) == ["a", "b", "c"]
        assert le.transform(np.array(["a", "b", "c"])).tolist() == [0, 1, 2]
