import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

data = pd.read_csv("C:/AI_With_Python/AI-with-Python/Assignment-5/Auto.csv")



y = pd.to_numeric(df['mpg'], errors='coerce')
X = df.drop(columns=['mpg', 'name', 'origin'])
X = X.apply(pd.to_numeric, errors='coerce')


data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns=['mpg']), data['mpg']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


alphas = [0.01, 0.1, 1, 10, 100]   
ridge_scores = []
lasso_scores = []

for a in alphas:
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    ridge_scores.append(ridge.score(X_test_scaled, y_test))
    
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_scores.append(lasso.score(X_test_scaled, y_test))


best_ridge_idx = np.argmax(ridge_scores)
best_lasso_idx = np.argmax(lasso_scores)

print(f"Best Ridge alpha: {alphas[best_ridge_idx]}, R² = {ridge_scores[best_ridge_idx]:.4f}")
print(f"Best Lasso alpha: {alphas[best_lasso_idx]}, R² = {lasso_scores[best_lasso_idx]:.4f}")


plt.plot(alphas, ridge_scores, marker='o', label='Ridge R²')
plt.plot(alphas, lasso_scores, marker='s', label='Lasso R²')
plt.xscale('log')
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² on test set")
plt.title("R² vs Alpha")
plt.legend()
plt.grid(True)
plt.show()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_ridge = Ridge(alpha=alphas[best_ridge_idx])
best_ridge.fit(X_train_scaled, y_train)

best_lasso = Lasso(alpha=alphas[best_lasso_idx], max_iter=10000)
best_lasso.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Ridge Coef': best_ridge.coef_,
    'Lasso Coef': best_lasso.coef_
})
print("\nCoefficients:")
print(coef_df)