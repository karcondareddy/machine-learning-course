import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

def minkowski_distance(x, y, p):

    x = np.array(x)
    y = np.array(y)

    return np.sum(np.abs(x - y) ** p) ** (1 / p)

vector1 = df.iloc[0].values
vector2 = df.iloc[1].values

distances = []

for p in range(1, 11):
    distances.append(minkowski_distance(vector1, vector2, p))

plt.plot(range(1, 11), distances, marker="o")
plt.title("Minkowski Distance")
plt.xlabel("p")
plt.ylabel("Distance")
plt.grid(True)
plt.show()