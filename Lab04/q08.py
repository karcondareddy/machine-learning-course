import pandas as pd
import numpy as np

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


def calculate_standard_deviation(values):
    variance = calculate_variance(values)

    return variance ** 0.5


def calculate_statistics(data):
    numeric_data = data.select_dtypes(include=np.number)

    means = {}
    variances = {}
    standard_deviations = {}

    for column in numeric_data.columns:
        values = numeric_data[column].dropna().values

        means[column] = calculate_mean(values)
        variances[column] = calculate_variance(values)
        standard_deviations[column] = calculate_standard_deviation(values)

    return means, variances, standard_deviations


means, variances, standard_deviations = calculate_statistics(data)

print("Mean")
print("-" * 40)

for column, value in means.items():
    print(f"{column:25} : {value:.4f}")


print("\nVariance")
print("-" * 40)

for column, value in variances.items():
    print(f"{column:25} : {value:.4f}")


print("\nStandard Deviation")
print("-" * 40)

for column, value in standard_deviations.items():
    print(f"{column:25} : {value:.4f}")