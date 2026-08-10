import numpy as np

def calculate_mean(data):

    total = 0

    for value in data:
        total += value

    return total / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    total = 0

    for value in data:
        total += (value - mean) ** 2

    return total / len(data)


data = np.array([
    10, 20, 30, 40, 50
])

result = calculate_variance(data)

print(
    "Variance:",
    result
)