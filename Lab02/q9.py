import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

df = df.replace({
    't': 1,
    'f': 0,
    'Yes': 1,
    'No': 0
}).infer_objects(copy=False)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.median(numeric_only=True))

numerical_columns = df.select_dtypes(include=np.number).columns

for col in numerical_columns:

    minimum = df[col].min()
    maximum = df[col].max()

    df[col] = (df[col] - minimum) / (maximum - minimum)

print(df.head())