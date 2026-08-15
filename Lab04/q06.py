import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")

def my_minkowski_distance(A, B, p):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)

numeric_data = data.select_dtypes(include=np.number)

A = numeric_data.iloc[0].values
B = numeric_data.iloc[1].values

print("Comparison of Minkowski Distance")
print("-" * 70)
print("p\tMy Function\t\tSciPy Function\t\tDifference")
print("-" * 70)

for p in range(1, 11):

    my_distance = my_minkowski_distance(A, B, p)

    scipy_distance = minkowski(A, B, p=p)

    difference = abs(my_distance - scipy_distance)

    print(
        f"{p}\t"
        f"{my_distance:.6f}\t\t"
        f"{scipy_distance:.6f}\t\t"
        f"{difference:.6f}"
    )