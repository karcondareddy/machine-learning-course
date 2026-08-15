import pandas as pd

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


def label_encode(data, column):
    values = data[column].unique()
    encoding = {value: index for index, value in enumerate(values)}
    data[column] = data[column].map(encoding)

    return data


def one_hot_encode(data, column):
    return pd.get_dummies(data, columns=[column], dtype=int)


categorical_columns = data.select_dtypes(include=["object"]).columns

print("Original Dataset Dimensions:")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

label_data = data.copy()

for column in categorical_columns:
    label_data = label_encode(label_data, column)

print("\nAfter Label Encoding:")
print("Rows    :", label_data.shape[0])
print("Columns :", label_data.shape[1])

one_hot_data = data.copy()

for column in categorical_columns:
    one_hot_data = one_hot_encode(one_hot_data, column)

print("\nAfter One-Hot Encoding:")
print("Rows    :", one_hot_data.shape[0])
print("Columns :", one_hot_data.shape[1])