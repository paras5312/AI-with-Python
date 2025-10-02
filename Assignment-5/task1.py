from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

results = {}
for features in [["bmi","s5"], ["bmi","s5","bp"], list(X.columns)]:
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=5)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[str(features)] = r2
    print(features, "MSE=%.2f" % mse, "R2=%.3f" % r2)

plt.bar(results.keys(), results.values(), color=['blue','green','red'])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=20)
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("50_Startups.csv")

print("Columns:", df.columns.tolist())
print("\nCorrelation:\n", df.corr(numeric_only=True))

X = df[["R&D Spend", "Marketing Spend"]]
y = df["Profit"]

plt.scatter(df["R&D Spend"], y); plt.title("R&D Spend vs Profit"); plt.show()
plt.scatter(df["Marketing Spend"], y); plt.title("Marketing Spend vs Profit"); plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)

print("Train RMSE:", np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
print("Train R²:", r2_score(y_train, model.predict(X_train)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
print("Test R²:", r2_score(y_test, model.predict(X_test)))