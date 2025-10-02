import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("weight-height.csv")

print(data.head())

X_height = data["Height"].values.reshape(-1, 1)
y_weight = data["Weight"].values

model = LinearRegression()
model.fit(X_height, y_weight)

y_pred = model.predict(X_height)

plt.scatter(X_height, y_weight, color="green", alpha=0.5, label="Actual points")
plt.plot(X_height, y_pred, color="red", label="Regression line")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.title("Height vs Weight: Linear Regression Model")
plt.legend()
plt.show()

rmse_score = np.sqrt(mean_squared_error(y_weight, y_pred))
r2_score_val = r2_score(y_weight, y_pred)

print("RMSE:", rmse_score)
print("R^2:", r2_score_val)

print("The results suggest that height and weight are positively correlated. RMSE gives the average error in prediction, and R^2 shows how much of weight variation can be explained by height.")
