import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

df = df.replace({'t': 1, 'f': 0, 'Yes': 1, 'No': 0})
dff = df.copy()

numerical_columns = dff.select_dtypes(include=np.number).columns
categorical_columns = dff.select_dtypes(exclude=np.number).columns

for col in numerical_columns:

    Q1 = dff[col].quantile(0.25)
    Q3 = dff[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = dff[(dff[col] < lower) | (dff[col] > upper)]

    if len(outliers) > 0:
        dff[col] = dff[col].fillna(dff[col].median())
    else:
        dff[col] = dff[col].fillna(dff[col].mean())

for col in categorical_columns:
    dff[col] = dff[col].fillna(dff[col].mode()[0])

print(dff.isnull().sum())
print("\nMissing values filled successfully.")