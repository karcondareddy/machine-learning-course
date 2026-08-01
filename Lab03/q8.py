import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

data = df.select_dtypes(include=np.number)

def my_mean(values):
    total = 0
    n = len(values)

    for value in values:
        total += value

    return total / n

def my_variance(values):
    mean = my_mean(values)

    total = 0

    for value in values:
        total += (value - mean) ** 2

    return total / len(values)

def my_std(values):
    return my_variance(values) ** 0.5


print("Feature Statistics\n")

for column in data.columns:

    values = data[column].values

    print(column)

    print("Mean      :", my_mean(values))
    print("Variance  :", my_variance(values))
    print("Std Dev   :", my_std(values))
    print()