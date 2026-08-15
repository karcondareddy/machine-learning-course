import pandas as pd

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


def identify_feature_types(data):
    feature_types = {}

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            feature_types[column] = "Interval"
        elif pd.api.types.is_numeric_dtype(data[column]):
            feature_types[column] = "Ratio"
        else:
            feature_types[column] = "Nominal"

    return feature_types

feature_types = identify_feature_types(data)

print("Feature\t\t\tType")
print("-" * 45)

for feature, feature_type in feature_types.items():
    print(f"{feature:25} {feature_type}")