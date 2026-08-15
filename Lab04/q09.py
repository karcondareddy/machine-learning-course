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
    return calculate_variance(values) ** 0.5

numeric_data = data.select_dtypes(include=np.number)

print("Comparison of Mean and Standard Deviation")
print("-" * 100)
print(f"{'Feature':35} {'My Mean':15} {'NumPy Mean':15} "
      f"{'My Std':15} {'NumPy Std':15}")
print("-" * 100)

for column in numeric_data.columns:

    values = numeric_data[column].dropna().values

    my_mean = calculate_mean(values)
    my_std = calculate_standard_deviation(values)

    numpy_mean = np.mean(values)
    numpy_std = np.std(values)

    print(
        f"{column:30} "
        f"{my_mean:15.4f} "
        f"{numpy_mean:15.4f} "
        f"{my_std:15.4f} "
        f"{numpy_std:15.4f}"
    )