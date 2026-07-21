import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
dff = df.replace({'t': 1, 'f': 0})
dff = dff.replace({'Yes': 1, 'No': 0})

binary_columns = []

for col in dff.columns:
    unique_values = dff[col].dropna().unique()

    if len(unique_values) == 2:
        binary_columns.append(col)

print("Binary Columns:")
print(binary_columns)

v1 = dff.loc[0, binary_columns]
v2 = dff.loc[1, binary_columns]

f11 = 0
f10 = 0
f01 = 0
f00 = 0

for a, b in zip(v1, v2):

    if a == 1 and b == 1:
        f11 += 1

    elif a == 1 and b == 0:
        f10 += 1

    elif a == 0 and b == 1:
        f01 += 1

    elif a == 0 and b == 0:
        f00 += 1

print(f11, f10, f01, f00)

JC = f11 / (f11 + f10 + f01)

SMC = (f11 + f00) / (f11 + f10 + f01 + f00)

print("\nf11 =", f11)
print("f10 =", f10)
print("f01 =", f01)
print("f00 =", f00)

print("\nJaccard Coefficient =", JC)
print("Simple Matching Coefficient =", SMC)