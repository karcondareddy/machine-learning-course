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


def calculate_standard_deviation(data):

    variance = calculate_variance(data)

    return np.sqrt(variance)


data = np.array([
    10, 20, 30, 40, 50
])

result = calculate_standard_deviation(data)

print(
    "Standard Deviation:",
    result
)