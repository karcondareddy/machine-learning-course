import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

df = df.replace({
    't': 1,
    'f': 0,
    'Yes': 1,
    'No': 0
}).infer_objects(copy=False)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(0)

data = df.iloc[:20]

n = len(data)

JC = np.zeros((n, n))
SMC = np.zeros((n, n))
COS = np.zeros((n, n))

for i in range(n):

    v1 = data.iloc[i].to_numpy(dtype=float)

    for j in range(n):

        v2 = data.iloc[j].to_numpy(dtype=float)

        # ---------- Jaccard and SMC ----------
        f11 = f10 = f01 = f00 = 0

        for a, b in zip(v1, v2):

            a = 1 if a != 0 else 0
            b = 1 if b != 0 else 0

            if a == 1 and b == 1:
                f11 += 1

            elif a == 1 and b == 0:
                f10 += 1

            elif a == 0 and b == 1:
                f01 += 1

            else:
                f00 += 1

        denominator = f11 + f10 + f01

        if denominator == 0:
            JC[i][j] = 0
        else:
            JC[i][j] = f11 / denominator

        SMC[i][j] = (f11 + f00) / (f11 + f10 + f01 + f00)

        # ---------- Cosine Similarity ----------
        dot = np.dot(v1, v2)

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            COS[i][j] = 0
        else:
            COS[i][j] = dot / (mag1 * mag2)

plt.figure(figsize=(8,6))
sns.heatmap(JC, annot=True, cmap="Blues")
plt.title("Jaccard Coefficient Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(SMC, annot=True, cmap="Greens")
plt.title("Simple Matching Coefficient Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(COS, annot=True, cmap="Reds")
plt.title("Cosine Similarity Heatmap")
plt.show()