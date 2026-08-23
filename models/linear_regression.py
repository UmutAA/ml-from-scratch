from metrics.metrics import calculate_mse
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel():
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
        self.b = 0
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        #Reshape the inputs matrix as column matrix
        if X.ndim == 1:
            X = X.reshape(-1,1)

        m,n = X.shape #m row n columns: m samples, n features
        self.w = np.zeros(n, dtype=float)

        #Training loop
        for epoch in range(self.epochs):
            #Matrix multiplication (m,n) @ (n,1) -> (m,1)
            y_preds = np.dot(X, self.w) + self.b

            #Calculating loss using mse function
            loss = calculate_mse(y, y_preds)
            self.loss_history.append(loss)

            #Adjusting parameters

            #Calculating gradients
            dw = (-2.0 / m) * np.dot(X.T, (y - y_preds))
            db = (-2.0 / m) * np.sum((y - y_preds))

            #Gradient Descent
            self.w -= self.lr * dw
            self.b -= self.lr * db

            #Print the current state every 100 epoch
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss:}")

            if loss == 0:
                break

    def predict(self, X: np.ndarray | int):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return (np.dot(X, self.w) + self.b)

    def plot(self, X: np.ndarray, y: np.ndarray):
        if X.ndim > 1 and X.shape[1] > 1:
            print("Can't plot 2D plots of multivariate values")
            return

        plt.scatter(X, y, c="green", label="Observed Values")
        y_preds = self.predict(X)
        plt.plot(X, y_preds, color="red", label="Predicted Values")
        plt.legend()
        plt.title(f"Linear Regression Model")
        plt.show()

    def print_formula(self):
        if self.w is None:
            raise RuntimeError("Model has not been fitted yet. Please call 'fit' method first.")

        weights = np.atleast_1d(self.w)
        formula_terms = [f"{w_val:+.4f} * x{i+1}" for i, w_val in enumerate(weights)]
        formula = " ".join(formula_terms) + f" {self.b:+.4f}"
        if formula.startswith("+ "):
            formula = formula[2:]
            
        print(f"Learned formula: \ny = {formula}")