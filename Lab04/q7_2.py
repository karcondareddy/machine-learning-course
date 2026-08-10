import numpy as np

def calculate_euclidean_norm(A):

    sum_of_squares = 0

    for value in A:
        sum_of_squares += value ** 2

    return np.sqrt(
        sum_of_squares
    )


A = np.array([3, 4])

own_result = calculate_euclidean_norm(A)

numpy_result = np.linalg.norm(A)

print(
    "My Euclidean Norm:",
    own_result
)

print(
    "NumPy Euclidean Norm:",
    numpy_result
)