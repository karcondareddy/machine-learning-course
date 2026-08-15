import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)


def dot_product(A, B):
    total = 0

    for i in range(len(A)):
        total = total + A[i] * B[i]

    return total


def vector_length(A):
    total = 0

    for value in A:
        total = total + value ** 2

    return total ** 0.5


data = df.select_dtypes(include=np.number).dropna()

A = data.iloc[0].values
B = data.iloc[1].values


print("My Dot Product:")
print(dot_product(A, B))

print("\nNumPy Dot Product:")
print(np.dot(A, B))


print("\nMy Euclidean Norm:")
print(vector_length(A))

print("\nNumPy Euclidean Norm:")
print(np.linalg.norm(A))