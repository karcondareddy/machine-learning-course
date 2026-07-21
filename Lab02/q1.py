import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()

y = df["Payment (Rs)"].to_numpy().reshape(-1, 1)

print("Feature Matrix (X):")
print(X)

print("\nOutput Vector (y):")
print(y)

print("\nDimensionality of Feature Matrix:", X.shape[1])

print("Number of Observation Vectors:", X.shape[0])

print("\n Dimension:", X.shape)

rank = np.linalg.matrix_rank(X)

print("\nRank of Feature Matrix:", rank)

X_pinv = np.linalg.pinv(X)

print("\nPseudo Inverse of X:")
print(X_pinv)

cost = X_pinv @ y
print(cost)

print("\nCost of each product:")

print("Candy  :", cost[0][0])
print("Mango  :", cost[1][0])
print("Milk   :", cost[2][0])