import unittest
import numpy as np

# GenAI Tool Used: ChatGPT


def minkowski_distance(A, B, p=2):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)


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


def calculate_mean(values):
    total = 0

    for value in values:
        total += value

    return total / len(values)


def calculate_variance(values):
    mean = calculate_mean(values)
    total = 0

    for value in values:
        total += (value - mean) ** 2

    return total / len(values)


def calculate_standard_deviation(values):
    return calculate_variance(values) ** 0.5


class TestLabFunctions(unittest.TestCase):

    # Test Minkowski distance
    def test_minkowski_distance(self):
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])

        self.assertAlmostEqual(
            minkowski_distance(A, B, 2),
            np.sqrt(27)
        )

    # Test Dot Product
    def test_dot_product(self):
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])

        self.assertEqual(
            dot_product(A, B),
            32
        )

    # Test Euclidean Norm
    def test_euclidean_norm(self):
        A = np.array([3, 4])

        self.assertEqual(
            euclidean_norm(A),
            5
        )

    # Test Mean
    def test_mean(self):
        values = [10, 20, 30, 40]

        self.assertEqual(
            calculate_mean(values),
            25
        )

    # Test Variance
    def test_variance(self):
        values = [10, 20, 30, 40]

        self.assertEqual(
            calculate_variance(values),
            125
        )

    # Test Standard Deviation
    def test_standard_deviation(self):
        values = [10, 20, 30, 40]

        self.assertAlmostEqual(
            calculate_standard_deviation(values),
            np.sqrt(125)
        )


if __name__ == "__main__":
    unittest.main()