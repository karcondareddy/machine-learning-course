import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

def minkowski_distance(x, y, p):

    x = np.array(x)
    y = np.array(y)

    distance = np.sum(np.abs(x - y) ** p) ** (1 / p)

    return distance

vector1 = df.iloc[0].values
vector2 = df.iloc[1].values

print(minkowski_distance(vector1, vector2, 2))