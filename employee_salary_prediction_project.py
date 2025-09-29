# Requirement - 
# Predict an employee salary based on their years of experience 
# using Linear Regression, Python & Matplotlib

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
         "Experience": [1,2,3,4,5],
         "Salary": [30000,35000,40000,45000,50000]
       }

df = pd.DataFrame(data)


X = df[["Experience"]]
y = df[["Salary"]]

model = LinearRegression()

model.fit(X, y)

pred = model.predict([[6]])[0]
print(f"Predicted Salary : {pred}")

plt.scatter(X,y, color='blue')
plt.plot(X,model.predict(X),color='red')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()
