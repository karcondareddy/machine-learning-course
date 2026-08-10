import numpy as np

def minkowski_distance(A, B, p):

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if len(A) != len(B):
        raise ValueError(
            "Vectors must have the same dimensions."
        )

    if p <= 0:
        raise ValueError(
            "Order p must be greater than zero."
        )

    distance = (
        np.sum(np.abs(A - B) ** p)
    ) ** (1 / p)

    return distance


A = np.array([1, 2, 3])
B = np.array([4, 6, 8])

print(
    "Manhattan Distance:",
    minkowski_distance(A, B, 1)
)

print(
    "Euclidean Distance:",
    minkowski_distance(A, B, 2)
)