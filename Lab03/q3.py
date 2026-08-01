import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

# Label Encoding Function
def my_label_encoder(series):
    unique_values = sorted(series.dropna().unique())

    mapping = {}
    for i, value in enumerate(unique_values):
        mapping[value] = i

    encoded = series.map(mapping)

    return encoded, mapping


# One Hot Encoding Function
def my_one_hot_encoder(series):

    unique_values = sorted(series.dropna().unique())

    encoded_df = pd.DataFrame()

    for value in unique_values:
        encoded_df[f"{series.name}_{value}"] = (series == value).astype(int)

    return encoded_df


categorical_columns = df.select_dtypes(include=["object"]).columns

print("Categorical Columns:")
print(categorical_columns)

# Label Encoding
label_df = df.copy()

for col in categorical_columns:
    label_df[col], mapping = my_label_encoder(label_df[col])
    print(f"\nMapping for {col}")
    print(mapping)

print("\nShape After Label Encoding:")
print(label_df.shape)

# One Hot Encoding
onehot_df = df.copy()

for col in categorical_columns:

    encoded = my_one_hot_encoder(onehot_df[col])

    onehot_df = onehot_df.drop(col, axis=1)

    onehot_df = pd.concat([onehot_df, encoded], axis=1)

print("\nShape After One Hot Encoding:")
print(onehot_df.shape)