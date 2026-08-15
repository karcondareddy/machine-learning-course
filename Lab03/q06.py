import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]


def minkowski_distance(x, y, p):

    x = np.array(x)
    y = np.array(y)

    return np.sum(np.abs(x - y) ** p) ** (1 / p)


vector1 = df.iloc[0].values
vector2 = df.iloc[1].values

for p in range(1, 11):

    my_distance = minkowski_distance(vector1, vector2, p)

    scipy_distance = minkowski(vector1, vector2, p)

    print(f"p = {p}")

    print("My Function :", my_distance)

    print("Scipy       :", scipy_distance)

    print()