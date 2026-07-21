import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

print("First Five Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

categorical_columns = df.select_dtypes(include=['object']).columns

numerical_columns = df.select_dtypes(include=[np.number]).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))

print("\nEncoding Suggestions")

for col in categorical_columns:
    print(col, "-> One-Hot Encoding (Nominal)")

print("\nData Range")

for col in numerical_columns:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

print("\nMissing Values")

print(df.isnull().sum())

print("\nMean")

print(df[numerical_columns].mean())

print("\nVariance")

print(df[numerical_columns].var())

print("\nOutlier Detection")

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col} : {len(outliers)} Outliers")