import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

df = pd.read_csv("data/churn.csv")

# feature extaction and encoding
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

df["Gender"] = (df["Gender"] == "Male").astype(int)
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# encode remaining categorical
if "Card Type" in df.columns:
    df = pd.get_dummies(df, columns=["Card Type"], drop_first=True)

df = df.astype(float)

y = df["Exited"].values
X = df.drop(columns=["Exited"]).values

# standardize
X_mean = np.mean(X, axis = 0)
X_std = np.std(X, axis = 0)
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

model = LogisticRegression(learning_rate=0.001, n_epochs=1000)
model.fit(X_train, y_train)

# Evaluating
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("accuracy:", accuracy)
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

z = X_test @ model.weights + model.bias
proba = 1 / (1 + np.exp(-z))

import matplotlib.pyplot as plt
import ipywidgets as widgets

feature_names = df.drop(columns=["Exited"]).columns.tolist()

def plot_feature(idx):
    plt.figure()
    plt.scatter(X_test[:, idx], proba, alpha=0.3)
    plt.xlabel(feature_names[idx])
    plt.ylabel("Predicted churn probability")
    plt.title(f"Probability vs {feature_names[idx]}")
    plt.show()

plot_feature(0)
