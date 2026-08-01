import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

data = df.select_dtypes(include=np.number)

def my_mean(values):
    return sum(values) / len(values)

def my_variance(values):
    mean = my_mean(values)

    total = 0

    for value in values:
        total += (value - mean) ** 2

    return total / len(values)

def my_std(values):
    return my_variance(values) ** 0.5

print("Comparison\n")

for column in data.columns:

    values = data[column].values

    print(column)

    print("Own Mean      :", my_mean(values))
    print("NumPy Mean    :", np.mean(values))

    print("Own Std Dev   :", my_std(values))
    print("NumPy Std Dev :", np.std(values))

    print()