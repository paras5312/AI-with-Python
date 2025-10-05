import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("C:/AI_With_Python/AI-with-Python/Assignment-5/a50_Startups.csv")


print("Columns:", data.columns.tolist())

X = data[['R&D Spend', 'Marketing Spend']]   
y = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.4f}")

plt.figure(figsize=(6,5))
plt.scatter(X_test['R&D Spend'], y_test, color='blue', label='Actual')
plt.scatter(X_test['R&D Spend'], y_pred, color='red', label='Predicted')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Actual vs Predicted Profit")
plt.legend()
plt.show()