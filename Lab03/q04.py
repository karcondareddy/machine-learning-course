import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)


def minkowski_distance(A, B, p):
    total = 0

    for i in range(len(A)):
        total = total + abs(A[i] - B[i]) ** p

    distance = total ** (1 / p)

    return distance


data = df.select_dtypes(include=np.number).dropna()

A = data.iloc[0].values
B = data.iloc[1].values


print("Manhattan Distance:")
print(minkowski_distance(A, B, 1))

print("\nEuclidean Distance:")
print(minkowski_distance(A, B, 2))