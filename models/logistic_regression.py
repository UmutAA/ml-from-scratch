from metrics.metrics import calculate_bce, sigmoid
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel():
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, print_rate: int = 1000):
            self.lr = learning_rate
            self.epochs = epochs
            self.loss_history = []
            self.b = None
            self.w = None
            self.print_rate = print_rate

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

            #Calculating loss using mse function
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

    def plot(self, X: np.ndarray, y: np.ndarray):
        if X.ndim > 1 and X.shape[1] > 1:
            print("Can't plot 2D plots of multivariate values")
            return
        
        # 1. Plot actual classes (observed values) directly
        plt.scatter(X, y, c="green", alpha=0.5, label="Actual Classes (Observed)", zorder=3)
        
        # 2. Sort X values to ensure a smooth continuous sigmoid curve plot
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx]
        
        z = np.dot(X_sorted, self.w) + self.b
        probabilities = sigmoid(z)
        
        plt.plot(X_sorted, probabilities, color="red", label="Predicted Probabilities (Sigmoid)", linewidth=2, zorder=2)
        
        # 3. Display discrete model classification predictions (0 or 1) on the curve
        y_pred_classes = self.predict(X_sorted, boolean=False)
        plt.scatter(X_sorted, y_pred_classes, c="blue", marker="x", alpha=0.7, label="Model Predictions (0 or 1)", zorder=4)
        
        # 4. Draw decision threshold boundary line at 0.5 probability
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
        formula_terms = [f"{w_val:+.4f} * x{i+1}" for i, w_val in enumerate(weights)]
        formula = " ".join(formula_terms) + f" {self.b:+.4f}"
        if formula.startswith("+ "):
            formula = formula[2:]
                
        print(f"Learned formula: \ny = {formula}")