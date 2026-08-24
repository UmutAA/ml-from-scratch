import pandas as pd
import numpy as np
from models.linear_regression import LinearRegressionModel
from preprocessing.preprocessing import StandardScaler
from metrics.metrics import calculate_mse


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
