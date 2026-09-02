import pandas as pd
import numpy as np
from models.linear_regression import LinearRegressionModel
from models.logistic_regression import LogisticRegressionModel
from preprocessing.preprocessing import StandardScaler

def generate_synthetic_data(n_samples: int = 100):
    """
    Generates a linearly separable synthetic dataset for testing.
    """
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(n_samples, 1))
    
    z = 2.5 * X.squeeze() + 0.5
    probabilities = 1 / (1 + np.exp(-z))
    y = np.array([1 if p > 0.5 else 0 for p in probabilities])
    
    return X, y

if __name__ == "__main__":

    #Linear Regression Model Test
    """
    model = LinearRegressionModel(learning_rate=0.001, epochs=1000000)

    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")

    test.dropna(axis=0)
    train.dropna(axis=0)

    X_train = train.drop("SalePrice", axis=1)
    y_train = train["SalePrice"]
    X_test = test.drop("SalePrice", axis=1)
    y_test = test["SalePrice"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    model.print_formula()
    preds = model.predict(X_test)

    preds = pd.DataFrame(preds)

    print(preds.head())

    """

    #Logistic Regression Model Test

    print("1. Generating synthetic dataset...")
    X_train, y_train = generate_synthetic_data(n_samples=150)
        
    print(f"Data shapes - X: {X_train.shape}, y: {y_train.shape}")
        
    print("\n2. Initializing and training LogisticRegressionModel...")
    model = LogisticRegressionModel(learning_rate=0.5, epochs=2000, print_rate=500)
    model.fit(X_train, y_train)
        
    print("\n3. Printing learned formula:")
    model.print_formula()
        
    print("\n4. Making predictions on test samples...")
    sample_test = np.array([[-1.5], [0.0], [1.5]])
    predictions = model.predict(sample_test, boolean=False)
        
    for val, pred in zip(sample_test.flatten(), predictions):
        print(f"Input: {val:+.1f} --> Predicted Class: {pred}")

    print("\n5. Plotting decision boundary/predictions...")
    model.plot(X_train, y_train)


