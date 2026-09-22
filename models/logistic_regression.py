from metrics.metrics import calculate_bce, calculate_cce, sigmoid, softmax
from preprocessing.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class LogisticRegressionModel(ABC):
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, print_rate: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
        self.b = None
        self.w = None
        self.print_rate = print_rate

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, boolean: bool = False):
        pass

    @abstractmethod
    def plot(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def print_formula(self):
        pass

class BinaryLogisticRegressionModel(LogisticRegressionModel):
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, print_rate: int = 1000):
            super().__init__(learning_rate, epochs, print_rate)

    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-8):

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if not isinstance(y, np.ndarray):
             y = np.asarray(y)

        #Reshape the inputs matrix as column matrix
        if X.ndim == 1:
            X = X.reshape(-1,1)

        self.b = np.mean(y)

        m,n = X.shape #m row n columns: m samples, n features
        self.w = np.zeros(n, dtype=float)
        prev_loss = None

        #Training Loop
        for epoch in range(self.epochs):
            #Matrix multiplication (m,n) @ (n,1) -> (m,1)
            z = np.dot(X, self.w) + self.b
            p = sigmoid(z)

            #Calculating loss using bce function
            loss = calculate_bce(y, p)
            self.loss_history.append(loss)

            #Adjusting parameters

            #Calculating gradients
            dw = (1.0 / m) * np.dot(X.T, p - y)
            db = (1.0 / m) * np.sum(p - y)

            #Gradient Descent
            self.w -= self.lr * dw
            self.b -= self.lr * db

            #Print the current state
            if (epoch + 1) % self.print_rate == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:.4f}")

            #Early stopping for better performance
            if prev_loss is not None and abs(prev_loss - loss) < tol:
                print(f"Converged at epoch {epoch + 1}, Loss: {loss:.4f}")
                break

            prev_loss = loss

    def predict(self, X: np.ndarray, boolean: bool = False):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1,1)

        z = np.dot(X, self.w) + self.b
        p = sigmoid(z)

        value = p >= 0.5

        if boolean:
            return value
        else:
            return value.astype(int)

    def plot(self, X: np.ndarray, y: np.ndarray, boolean: bool = False):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] > 1:
            print("Can't plot 2D plots of multivariate values")
            return

        # Plot actual classes (observed values) directly
        plt.scatter(X, y, c="green", alpha=0.5, label="Actual Classes (Observed)", zorder=3)

        # Sort X values to ensure a smooth continuous sigmoid curve plot
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]

        z = np.dot(X_sorted, self.w) + self.b
        probabilities = sigmoid(z)

        plt.plot(X_sorted, probabilities, color="red", label="Predicted Probabilities (Sigmoid)", linewidth=2, zorder=2)

        # Display discrete model classification predictions (0 or 1) on the curve
        y_pred_classes = self.predict(X_sorted, boolean=False)
        plt.scatter(X_sorted, y_pred_classes, c="blue", marker="x", alpha=0.7, label="Model Predictions (0 or 1)",
                    zorder=4)

        # Draw decision threshold boundary line at 0.5 probability
        plt.axhline(0.5, color="orange", linestyle="--", alpha=0.7, label="Decision Threshold (0.5)")

        plt.xlabel("X (Feature)")
        plt.ylabel("Probability / Class")
        plt.legend(loc="center left")
        plt.title("Logistic Regression: Observed vs Predicted")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()

    def print_formula(self):
        if self.w is None:
            raise RuntimeError("Model has not been fitted yet. Please call 'fit' method first.")

        weights = np.atleast_1d(self.w)
        formula_terms = [f"{w_val:+.4f} * x{i + 1}" for i, w_val in enumerate(weights)]
        formula = " ".join(formula_terms) + f" {self.b:+.4f}"
        if formula.startswith("+ "):
            formula = formula[2:]

        print(f"Learned formula: \ny = {formula}")

class MultinomialLogisticRegressionModel(LogisticRegressionModel):
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, print_rate: int = 1000):
        super().__init__(learning_rate, epochs, print_rate)
        self.label_encoder = LabelEncoder()
        self.n_classes = None

    def _one_hot(self, y_encoded: np.ndarray, n_classes: int) -> np.ndarray:
        one_hot = np.zeros((y_encoded.shape[0], n_classes))
        one_hot[np.arange(y_encoded.shape[0]), y_encoded] = 1
        return one_hot

    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-8):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = self.label_encoder.n_classes_
        Y_onehot = self._one_hot(y_encoded, self.n_classes)

        m, n = X.shape
        self.w = np.zeros((n, self.n_classes), dtype=float)
        self.b = np.zeros(self.n_classes, dtype=float)
        prev_loss = None

        for epoch in range(self.epochs):
            Z = np.dot(X, self.w) + self.b
            P = softmax(Z)

            loss = calculate_cce(Y_onehot, P)
            self.loss_history.append(loss)

            dw = (1.0 / m) * np.dot(X.T, P - Y_onehot)
            db = (1.0 / m) * np.sum(P - Y_onehot, axis=0)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (epoch + 1) % self.print_rate == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:.4f}")

            if prev_loss is not None and abs(prev_loss - loss) < tol:
                print(f"Converged at epoch {epoch + 1}, Loss: {loss:.4f}")
                break

            prev_loss = loss

    def predict(self, X: np.ndarray, boolean: bool = False):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Z = np.dot(X, self.w) + self.b
        P = softmax(Z)
        class_indices = np.argmax(P, axis=1)
        return self.label_encoder.classes_[class_indices]

    def plot(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] > 1:
            print("Can't plot 2D plots of multivariate values")
            return

        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]

        Z = np.dot(X_sorted, self.w) + self.b
        P = softmax(Z)  # (m, n_classes)

        y_encoded = self.label_encoder.transform(y)
        plt.scatter(X, y_encoded, c="green", alpha=0.5, label="Actual Classes (Observed)", zorder=3)

        for k in range(self.n_classes):
            plt.plot(X_sorted, P[:, k], linewidth=2,
                     label=f"P(class={self.label_encoder.classes_[k]})", zorder=2)

        y_pred_classes = self.predict(X_sorted)
        y_pred_encoded = self.label_encoder.transform(y_pred_classes)
        plt.scatter(X_sorted, y_pred_encoded, c="blue", marker="x", alpha=0.7,
                    label="Model Predictions", zorder=4)

        plt.xlabel("X (Feature)")
        plt.ylabel("Probability / Class")
        plt.legend(loc="center left")
        plt.title("Multinomial Logistic Regression: Observed vs Predicted")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()

    def print_formula(self):
        if self.w is None:
            raise RuntimeError("Model has not been fitted yet. Please call 'fit' method first.")

        for k in range(self.n_classes):
            class_label = self.label_encoder.classes_[k]
            weights = self.w[:, k]
            formula_terms = [f"{w_val:+.4f} * x{i + 1}" for i, w_val in enumerate(weights)]
            formula = " ".join(formula_terms) + f" {self.b[k]:+.4f}"
            if formula.startswith("+ "):
                formula = formula[2:]
            print(f"Learned formula (class={class_label}): \nz = {formula}")