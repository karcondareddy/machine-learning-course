import pandas as pd
import numpy as np

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


def dot_product(A, B):
    result = 0

    for i in range(len(A)):
        result += A[i] * B[i]

    return result


def euclidean_norm(A):
    result = 0

    for value in A:
        result += value ** 2

    return result ** 0.5

numeric_data = data.select_dtypes(include=np.number)

A = numeric_data.iloc[0].values
B = numeric_data.iloc[1].values

my_dot = dot_product(A, B)
numpy_dot = np.dot(A, B)

my_norm_A = euclidean_norm(A)
numpy_norm_A = np.linalg.norm(A)

my_norm_B = euclidean_norm(B)
numpy_norm_B = np.linalg.norm(B)


print("Dot Product")
print("-" * 40)
print("My Function :", my_dot)
print("NumPy       :", numpy_dot)
print("Difference  :", abs(my_dot - numpy_dot))

print("\nEuclidean Norm of A")
print("-" * 40)
print("My Function :", my_norm_A)
print("NumPy       :", numpy_norm_A)
print("Difference  :", abs(my_norm_A - numpy_norm_A))

print("\nEuclidean Norm of B")
print("-" * 40)
print("My Function :", my_norm_B)
print("NumPy       :", numpy_norm_B)
print("Difference  :", abs(my_norm_B - numpy_norm_B))