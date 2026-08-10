import numpy as np

def calculate_mean(data):
    total = 0

    for value in data:
        total += value

    return total / len(data)


data = np.array([
    10, 20, 30, 40, 50
])

result = calculate_mean(data)

print(
    "Mean:",
    result
)