import numpy as np
from scipy.spatial.distance import minkowski

def my_minkowski_distance(A, B, p):

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    return (
        np.sum(np.abs(A - B) ** p)
    ) ** (1 / p)


A = np.array([1, 2, 3, 4])
B = np.array([5, 6, 7, 8])


print(
    "p\tMy Function\tSciPy\tDifference"
)

for p in range(1, 11):

    own_result = my_minkowski_distance(
        A,
        B,
        p
    )

    scipy_result = minkowski(
        A,
        B,
        p=p
    )

    difference = abs(
        own_result - scipy_result
    )

    print(
        p,
        "\t",
        own_result,
        "\t",
        scipy_result,
        "\t",
        difference
    )