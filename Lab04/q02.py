import pandas as pd

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


def label_encode(data, column):
    values = data[column].unique()
    encoding = {value: index for index, value in enumerate(values)}
    data[column] = data[column].map(encoding)

    return data


def one_hot_encode(data, column):
    encoded_data = pd.get_dummies(data, columns=[column], dtype=int)

    return encoded_data


label_data = data.copy()
label_data = label_encode(label_data, "Education")

print("After Label Encoding:")
print(label_data[["Education"]].head())


one_hot_data = data.copy()
one_hot_data = one_hot_encode(one_hot_data, "Marital_Status")

print("\nAfter One-Hot Encoding:")
print(one_hot_data.head())