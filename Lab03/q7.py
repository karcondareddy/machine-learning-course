import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

def my_dot(A, B):

    total = 0

    for i in range(len(A)):
        total += A[i] * B[i]

    return total

def my_norm(A):

    total = 0

    for value in A:
        total += value ** 2

    return total ** 0.5

A = df.iloc[0].values
B = df.iloc[1].values

print("Own Dot Product:", my_dot(A, B))
print("NumPy Dot Product:", np.dot(A, B))

print()

print("Own Norm (A):", my_norm(A))
print("NumPy Norm (A):", np.linalg.norm(A))

print()

print("Own Norm (B):", my_norm(B))
print("NumPy Norm (B):", np.linalg.norm(B))