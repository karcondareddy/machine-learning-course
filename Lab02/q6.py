import pandas as pd
import numpy as np

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

df = df.replace({'t': 1, 'f': 0})
df = df.replace({'Yes': 1, 'No': 0})

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(0)

v1 = df.iloc[0].to_numpy(dtype=float)
v2 = df.iloc[1].to_numpy(dtype=float)

dot_product = np.dot(v1, v2)

magnitude_v1 = np.linalg.norm(v1)
magnitude_v2 = np.linalg.norm(v2)

cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2)

print("Dot Product =", dot_product)
print("Magnitude of Vector 1 =", magnitude_v1)
print("Magnitude of Vector 2 =", magnitude_v2)
print("Cosine Similarity =", cosine_similarity)