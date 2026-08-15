import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


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


values = data["Income"].dropna().values

mean = calculate_mean(values)
variance = calculate_variance(values)

print("Feature: Income")
print("Mean    :", mean)
print("Variance:", variance)

histogram, bins = np.histogram(values, bins=10)

print("\nHistogram Data")
print("-" * 40)

for i in range(len(histogram)):
    print(
        f"Range: {bins[i]:.2f} - {bins[i + 1]:.2f} "
        f"Count: {histogram[i]}"
    )

plt.hist(values, bins=10)

plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution")
plt.grid(True)

plt.show()