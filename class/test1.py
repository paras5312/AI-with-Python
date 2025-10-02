import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import inrain_test_split
from sklearn.linear_model import LinearRegression

svclassifier = SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
Y_pred = svclassifier.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))  