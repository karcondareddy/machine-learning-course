import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")

df = df.dropna()

feature1 = "MntWines"
feature2 = "Recency"
target = "Response"

X = df[[feature1, feature2]]
y = df[target]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

x1_min, x1_max = X[feature1].min() - 1, X[feature1].max() + 1
x2_min, x2_max = X[feature2].min() - 1, X[feature2].max() + 1

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))

grid = pd.DataFrame({feature1: xx1.ravel(), feature2: xx2.ravel()})
predictions = model.predict(grid)
predictions = predictions.reshape(xx1.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx1, xx2, predictions, alpha=0.3)
plt.scatter(X[feature1], X[feature2], c=y, edgecolors="black", s=30)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title("Decision Boundary using MntWines and Recency")
plt.show()