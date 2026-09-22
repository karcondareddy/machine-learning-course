import pandas as pd
import numpy as np


def calculate_gini_index(target_col):
    elements, counts = np.unique(target_col, return_counts=True)

    probabilities = counts / np.sum(counts)

    gini = 1 - np.sum(probabilities ** 2)

    return gini


df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
df = df.dropna()
target_col = "Response"
gini_value = calculate_gini_index(df[target_col])
print(f"Dataset Gini Index: {gini_value:.4f}")