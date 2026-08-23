import pandas as pd
import numpy as np
from models.linear_regression import LinearRegressionModel
from metrics.metrics import calculate_mse

data = {
    'SquareFeet': [1200, 1500, 1800, 2400, 3000, 800, 1100, 2200, 2600, 1900],
    'Rooms': [2, 3, 3, 4, 5, 1, 2, 4, 4, 3],
    'Age': [10, 5, 20, 2, 1, 30, 15, 8, 3, 12],
    'Price': [150000, 185000, 210000, 280000, 350000, 100000, 135000, 260000, 300000, 220000]
}

df = pd.DataFrame(data)


X = df[['SquareFeet', 'Rooms', 'Age']]
y = df['Price']

model = LinearRegressionModel(learning_rate=0.0000001, epochs=50000)

model.fit(X, y)
model.print_formula()

new_house = np.array([[2000, 3, 5]])
print("Prediction:", model.predict(new_house))
