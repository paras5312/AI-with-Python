import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("C:/AI_With_Python/AI-with-Python/Assignment-6/bank.csv", delimiter=";")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head(), "\n")

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print("Selected relevant columns for analysis.\n")

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)
print("Categorical variables converted to numeric dummy variables.\n")

plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

y = df3['y'].apply(lambda x: 1 if x == 'yes' else 0)  
X = df3.drop(columns=['y'])
print("Separated features (X) and target (y).\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized for KNN.\n")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("=== Logistic Regression Results ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n=== KNN (k=3) Results ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))