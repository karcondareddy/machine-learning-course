import pandas as pd
import numpy as np


def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)

    probabilities = counts / np.sum(counts)

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return entropy


df = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
df = df.dropna()
target_col = "Response"
entropy_value = calculate_entropy(df[target_col])
print(f"Dataset Entropy: {entropy_value:.4f}")