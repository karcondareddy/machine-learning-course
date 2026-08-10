import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minkowski_distance(A, B, p):

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    return (
        np.sum(np.abs(A - B) ** p)
    ) ** (1 / p)


file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

numeric_df = df.select_dtypes(
    include=np.number
).dropna()

A = numeric_df.iloc[0].values
B = numeric_df.iloc[1].values

p_values = []
distances = []

for p in range(1, 11):

    distance = minkowski_distance(
        A,
        B,
        p
    )

    p_values.append(p)
    distances.append(distance)

    print(
        "p =", p,
        "Distance =", distance
    )


plt.plot(
    p_values,
    distances,
    marker="o"
)

plt.xlabel("p")
plt.ylabel("Minkowski Distance")
plt.title(
    "Minkowski Distance for p = 1 to 10"
)

plt.grid(True)
plt.show()