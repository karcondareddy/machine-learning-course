import pandas as pd
import numpy as np

def calculate_entropy(target_col):
    if len(target_col) == 0:
        return 0

    elements, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / np.sum(counts)

    return -np.sum(probabilities * np.log2(probabilities))


def calculate_information_gain(data, split_feature, target_name):
    total_entropy = calculate_entropy(data[target_name])
    values, counts = np.unique(data[split_feature], return_counts=True)

    weighted_entropy = 0

    for i in range(len(values)):
        subset = data[data[split_feature] == values[i]]
        weight = counts[i] / np.sum(counts)
        weighted_entropy += weight * calculate_entropy(subset[target_name])

    return total_entropy - weighted_entropy


def find_root_node(data, feature_columns, target_name):
    information_gains = {}

    for feature in feature_columns:
        information_gains[feature] = calculate_information_gain(data, feature, target_name)

    return max(information_gains, key=information_gains.get)


def bin_feature(feature_data, bins=4, binning_type="width"):
    if binning_type == "width":
        return pd.cut(feature_data, bins=bins, labels=False)

    elif binning_type == "frequency":
        return pd.qcut(feature_data, q=bins, labels=False, duplicates="drop")

    else:
        raise ValueError("binning_type must be 'width' or 'frequency'")


def build_custom_decision_tree(data, feature_columns, target_name, depth=0, max_depth=3):
    if len(data) == 0:
        return None

    unique_targets = np.unique(data[target_name])

    if len(unique_targets) == 1:
        return unique_targets[0]

    if len(feature_columns) == 0 or depth >= max_depth:
        return data[target_name].mode()[0]

    best_feature = find_root_node(data, feature_columns, target_name)

    tree = {best_feature: {}}

    remaining_features = [feature for feature in feature_columns if feature != best_feature]

    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]

        tree[best_feature][value] = build_custom_decision_tree(
            subset,
            remaining_features,
            target_name,
            depth + 1,
            max_depth
        )

    return tree


def print_decision_tree(tree, level=0):
    if not isinstance(tree, dict):
        print("  " * level + "→ Prediction:", tree)
        return

    feature = next(iter(tree))

    print("  " * level + "Feature:", feature)

    for value, branch in tree[feature].items():
        print("  " * (level + 1) + f"Value = {value}")
        print_decision_tree(branch, level + 2)


df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")

df = df.dropna()

target_col = "Response"

# Remove ID because it is only an identifier
feature_cols = [col for col in df.columns if col not in [target_col, "ID"]]

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
df["Dt_Customer"] = (df["Dt_Customer"] - df["Dt_Customer"].min()).dt.days

for feature in feature_cols:
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > 4:
        df[feature] = bin_feature(df[feature], bins=4, binning_type="width")

custom_tree = build_custom_decision_tree(df, feature_cols, target_col, max_depth=3)

print("Custom Decision Tree:")
print_decision_tree(custom_tree)