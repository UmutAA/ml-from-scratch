import numpy as np

from metrics.metrics import classification_report, confusion_matrix
from models.linear_regression import LinearRegressionModel
from models.logistic_regression import LogisticRegressionModel
from preprocessing.preprocessing import StandardScaler, train_test_split


def generate_synthetic_data(n_samples: int = 100, noise_std: float = 0.5):
    """
    Generates a linearly separable synthetic dataset for testing.
    """
    np.random.seed(42)
    X = np.random.uniform(-2, 2, size=(n_samples, 1))

    noise = np.random.normal(0, noise_std, size=n_samples)
    z = 2.5 * X.squeeze() + 0.5 + noise

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
    X, y = generate_synthetic_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n2. Initializing and training LogisticRegressionModel...")
    model = LogisticRegressionModel(learning_rate=0.5, epochs=2000, print_rate=500)
    model.fit(X_train, y_train)
    print("\n3. Printing learned formula:")
    model.print_formula()

    print("\n4. Making predictions on the test set...")
    test_predictions = model.predict(X_test, boolean=False)

    print("\n5. Plotting decision boundary on the training data...")
    model.plot(X_test, y_test)

    print("\n6. Plotting confusion matrix...")
    cf = confusion_matrix(y_test, test_predictions)
    print(cf)

    print("\n7. Classification report on the test set:")
    report = classification_report(y_test, test_predictions)
    print(report)


