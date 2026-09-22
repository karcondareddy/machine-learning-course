import pandas as pd
import numpy as np

def bin_feature(feature_data, bins=4, binning_type='width'):

    if binning_type == 'width':
        return pd.cut(feature_data, bins=bins, labels=False)

    elif binning_type == 'frequency':
        return pd.qcut(feature_data, q=bins, labels=False, duplicates='drop')

    else:
        raise ValueError("binning_type must be 'width' or 'frequency'")

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")

df = df.dropna()

target_col = "Response"
feature_cols = list(df.columns.drop(target_col))

print("Original Data (First 3 rows):")
print(df[feature_cols].head(3))

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
        df[col] = bin_feature(df[col], bins=4, binning_type='width')

print("\nBinned Data (First 3 rows):")
print(df[feature_cols].head(3))