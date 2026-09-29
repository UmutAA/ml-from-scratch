"""
Runs every model in this repository against its scikit-learn counterpart,
on real datasets, and saves comparison plots under the 'assets' folder.

Usage:
    python main.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    Ridge as SkRidge,
    Lasso as SkLasso,
    LogisticRegression as SkLogisticRegression,
)

from models.linear_regression import LinearRegressionModel, Ridge, Lasso
from models.logistic_regression import (
    BinaryLogisticRegressionModel,
    MultinomialLogisticRegressionModel,
)
from preprocessing.preprocessing import StandardScaler, train_test_split
from metrics.metrics import (
    calculate_mse,
    calculate_rmse,
    calculate_bce,
    confusion_matrix,
    classification_report,
    sigmoid,
)

QUIET = 10 ** 9
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Data loading (all real datasets, split and scaled with this repo's own code)
# ---------------------------------------------------------------------------

def load_regression_data(seed=42):
    """Diabetes progression dataset (real patient data, from scikit-learn)."""
    np.random.seed(seed)
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def load_binary_classification_data(seed=42):
    """Breast cancer diagnostic dataset (real medical data, from scikit-learn)."""
    np.random.seed(seed)
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def load_multiclass_data(seed=42):
    """Iris dataset (real flower measurements, from scikit-learn)."""
    np.random.seed(seed)
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, data.target_names


# ---------------------------------------------------------------------------
# Model runners: this repo's model vs. the matching sklearn model
# ---------------------------------------------------------------------------

def run_linear_regression():
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION: this repo vs. scikit-learn")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names = load_regression_data()

    mine = LinearRegressionModel(learning_rate=0.1, epochs=5000, print_rate=QUIET)
    mine.fit(X_train, y_train, tol=1e-12)
    mine_preds = mine.predict(X_test)

    ref = SkLinearRegression().fit(X_train, y_train)
    ref_preds = ref.predict(X_test)

    print_comparison_row("MSE", calculate_mse(y_test, mine_preds), calculate_mse(y_test, ref_preds))
    print_comparison_row("RMSE", calculate_rmse(y_test, mine_preds), calculate_rmse(y_test, ref_preds))
    print_comparison_row("R2", r2_score(y_test, mine_preds), r2_score(y_test, ref_preds))
    print(f"Max weight difference: {np.max(np.abs(mine.w - ref.coef_)):.5f}")

    plot_prediction_comparison(
        y_test, mine_preds, ref_preds,
        title="Linear Regression: Predicted vs Actual",
        filename="linear_regression_vs_sklearn.png",
    )
    return mine, ref


def run_regularized_regression():
    print("\n" + "=" * 70)
    print("RIDGE (L2) AND LASSO (L1): this repo vs. scikit-learn")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names = load_regression_data()
    m = len(y_train)

    # Alpha mapping: this repo's loss is MSE + alpha * penalty, while sklearn's
    # Ridge uses sum-of-squared-errors + alpha_sk * ||w||^2 (alpha_sk = alpha * m)
    # and sklearn's Lasso uses (1 / 2m) * sum-of-squared-errors + alpha_sk * |w|
    # (alpha_sk = alpha / 2). Both are converted below so the two sides are
    # solving the same optimization problem.
    alpha = 0.05

    ridge = Ridge(learning_rate=0.1, epochs=5000, print_rate=QUIET, alpha=alpha)
    ridge.fit(X_train, y_train, tol=1e-12)
    ridge_ref = SkRidge(alpha=alpha * m).fit(X_train, y_train)

    lasso = Lasso(learning_rate=0.05, epochs=5000, print_rate=QUIET, alpha=alpha)
    lasso.fit(X_train, y_train, tol=1e-12)
    lasso_ref = SkLasso(alpha=alpha / 2, max_iter=50000, tol=1e-10).fit(X_train, y_train)

    for name, mine, ref in (("Ridge", ridge, ridge_ref), ("Lasso", lasso, lasso_ref)):
        mine_preds, ref_preds = mine.predict(X_test), ref.predict(X_test)
        print(f"\n{name}")
        print_comparison_row("  MSE", calculate_mse(y_test, mine_preds), calculate_mse(y_test, ref_preds))
        print_comparison_row("  R2", r2_score(y_test, mine_preds), r2_score(y_test, ref_preds))
        print(f"  Max weight difference: {np.max(np.abs(mine.w - ref.coef_)):.5f}")

    plot_weight_comparison(
        feature_names,
        {"Ridge (mine)": ridge, "Ridge (sklearn)": ridge_ref,
         "Lasso (mine)": lasso, "Lasso (sklearn)": lasso_ref},
        title="Ridge vs Lasso Learned Weights: this repo vs. scikit-learn",
        filename="ridge_lasso_vs_sklearn.png",
    )
    return ridge, lasso


def run_binary_logistic_regression():
    print("\n" + "=" * 70)
    print("BINARY LOGISTIC REGRESSION: this repo vs. scikit-learn")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names = load_binary_classification_data()

    mine = BinaryLogisticRegressionModel(learning_rate=0.1, epochs=3000, print_rate=QUIET)
    mine.fit(X_train, y_train, tol=1e-12)
    mine_preds = mine.predict(X_test)
    mine_probs = sigmoid(np.dot(X_test, mine.w) + mine.b)

    # sklearn's LogisticRegression applies L2 regularization by default, while
    # this repo's BinaryLogisticRegressionModel applies none. Left as sklearn's
    # default (rather than forcing regularization off) because the breast
    # cancer features are linearly separable once standardized: an
    # unregularized model's weights grow without bound chasing a perfect fit,
    # which is not a meaningful thing to match. Accuracy and the confusion
    # matrix are still directly comparable; raw weight magnitudes are not.
    ref = SkLogisticRegression(max_iter=5000).fit(X_train, y_train)
    ref_preds = ref.predict(X_test)
    # Clipped because predict_proba can return exact 0 or 1, which sends
    # calculate_bce's log(p) to -inf.
    ref_probs = np.clip(ref.predict_proba(X_test)[:, 1], 1e-12, 1 - 1e-12)

    mine_report = classification_report(y_test, mine_preds, output_dict=True)
    ref_report = classification_report(y_test, ref_preds, output_dict=True)

    print_comparison_row("BCE", calculate_bce(y_test, mine_probs), calculate_bce(y_test, ref_probs))
    print_comparison_row("Accuracy", mine_report["accuracy"], ref_report["accuracy"])
    print_comparison_row("Class 1 precision", mine_report["1"]["precision"], ref_report["1"]["precision"])
    print_comparison_row("Class 1 recall", mine_report["1"]["recall"], ref_report["1"]["recall"])
    print_comparison_row("Class 1 F1 score", mine_report["1"]["f1_score"], ref_report["1"]["f1_score"])
    print(f"Max |weight| (mine):    {np.max(np.abs(mine.w)):.4f}  (unregularized, expected to be large)")
    print(f"Max |weight| (sklearn): {np.max(np.abs(ref.coef_[0])):.4f}  (L2-regularized)")

    plot_confusion_matrix_pair(
        confusion_matrix(y_test, mine_preds), confusion_matrix(y_test, ref_preds),
        labels=["Malignant (0)", "Benign (1)"],
        titles=("This repo", "scikit-learn"),
        suptitle="Binary Logistic Regression Confusion Matrix",
        filename="binary_logistic_regression_vs_sklearn.png",
    )
    return mine, ref


def run_multinomial_logistic_regression():
    print("\n" + "=" * 70)
    print("MULTINOMIAL LOGISTIC REGRESSION: this repo vs. scikit-learn")
    print("=" * 70)
    X_train, X_test, y_train, y_test, class_names = load_multiclass_data()

    mine = MultinomialLogisticRegressionModel(learning_rate=0.1, epochs=3000, print_rate=QUIET)
    mine.fit(X_train, y_train, tol=1e-12)
    mine_preds = mine.predict(X_test)

    ref = SkLogisticRegression(max_iter=5000).fit(X_train, y_train)
    ref_preds = ref.predict(X_test)

    mine_acc = np.mean(mine_preds == y_test)
    ref_acc = np.mean(ref_preds == y_test)
    print_comparison_row("Accuracy", mine_acc, ref_acc)

    plot_confusion_matrix_pair(
        confusion_matrix(y_test, mine_preds), confusion_matrix(y_test, ref_preds),
        labels=list(class_names),
        titles=("This repo", "scikit-learn"),
        suptitle="Multinomial Logistic Regression Confusion Matrix",
        filename="multinomial_logistic_regression_vs_sklearn.png",
    )
    return mine, ref


def print_comparison_row(label, mine_value, ref_value):
    print(f"  {label:<20} mine: {mine_value:>9.4f}   sklearn: {ref_value:>9.4f}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_prediction_comparison(y_true, mine_preds, ref_preds, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    lims = [y_true.min(), y_true.max()]

    for ax, preds, label in zip(axes, (mine_preds, ref_preds), ("This repo", "scikit-learn")):
        ax.scatter(y_true, preds, alpha=0.6, edgecolor="k")
        ax.plot(lims, lims, color="red", linestyle="--", label="Perfect fit")
        ax.set_title(label)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, filename)


def plot_weight_comparison(feature_names, models: dict, title, filename):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(feature_names))
    width = 0.8 / len(models)
    for i, (name, model) in enumerate(models.items()):
        weights = model.w if hasattr(model, "w") else model.coef_
        ax.bar(x + i * width, weights, width=width, label=name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Weight value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


def plot_confusion_matrix_pair(matrix_mine, matrix_ref, labels, titles, suptitle, filename):
    n = matrix_mine.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(9 + 0.6 * n, 4 + 0.4 * n))
    for ax, matrix, title in zip(axes, (matrix_mine, matrix_ref), titles):
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
        ax.set_title(title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    _save(fig, filename)


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    run_linear_regression()
    run_regularized_regression()
    run_binary_logistic_regression()
    run_multinomial_logistic_regression()
    print("\nAll models trained and compared against scikit-learn. Plots saved under the 'assets' folder.")