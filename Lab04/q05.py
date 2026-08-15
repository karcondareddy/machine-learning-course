import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")

def minkowski_distance(A, B, p):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)

numeric_data = data.select_dtypes(include=np.number)

A = numeric_data.iloc[0].values
B = numeric_data.iloc[1].values

p_values = range(1, 11)
distances = []

for p in p_values:
    distance = minkowski_distance(A, B, p)
    distances.append(distance)

print("Minkowski Distance")
print("-" * 30)

for p, distance in zip(p_values, distances):
    print(f"p = {p:2}  Distance = {distance:.4f}")

plt.plot(p_values, distances, marker="o")

plt.xlabel("p")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance for p = 1 to 10")
plt.xticks(list(p_values))
plt.grid(True)

plt.show()