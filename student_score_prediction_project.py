import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("student.csv")
print(data)

X = data[['Hours']]  #double brackets - 2D input
y = data[['Score']]  #Target column

model = LinearRegression()
model.fit(X,y)

predicted_score = model.predict(X)
print("test: ", predicted_score)

#evaluate
mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

#show results
print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)

new_hour = float(input("enter a hour =  "))
print(new_hour)

new_pred = model.predict([[new_hour]])

print(f"Prediction for {new_hour} is score = {new_pred}")