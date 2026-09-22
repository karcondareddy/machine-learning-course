import pandas as pd
import numpy as np

def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_information_gain(data, split_feature, target_name):
    total_entropy = calculate_entropy(data[target_name])
    values, counts = np.unique(data[split_feature], return_counts=True)

    weighted_entropy = 0

    for i in range(len(values)):
        subset = data[data[split_feature] == values[i]]
        subset_entropy = calculate_entropy(subset[target_name])
        weight = counts[i] / np.sum(counts)
        weighted_entropy += weight * subset_entropy

    return total_entropy - weighted_entropy

def find_root_node(data, feature_columns, target_name):
    information_gains = {}

    for feature in feature_columns:
        information_gains[feature] = calculate_information_gain(data, feature, target_name)

    root_node = max(information_gains, key=information_gains.get)

    return root_node, information_gains

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")

df = df.dropna()

target_col = "Response"
feature_cols = list(df.columns.drop(target_col))

for feature in feature_cols:
    if pd.api.types.is_numeric_dtype(df[feature]):
        df[feature] = pd.cut(df[feature], bins=4, labels=False)
    else:
        df[feature] = df[feature].astype(str)

root_node, information_gains = find_root_node(df, feature_cols, target_col)

print(f"Selected Root Node Attribute: {root_node}")

print("\nInformation Gains:")

for feature, gain in information_gains.items():
    print(f"  {feature}: {gain:.4f}")