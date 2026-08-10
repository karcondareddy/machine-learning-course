import numpy as np

def calculate_dot_product(A, B):

    if len(A) != len(B):
        raise ValueError(
            "Vectors must have the same dimensions."
        )

    result = 0

    for i in range(len(A)):
        result += A[i] * B[i]

    return result


A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

own_result = calculate_dot_product(A, B)

numpy_result = np.dot(A, B)

print(
    "My Dot Product:",
    own_result
)

print(
    "NumPy Dot Product:",
    numpy_result
)