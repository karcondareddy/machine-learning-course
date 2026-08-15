import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

def label_encoding(data, column):

    values = data[column].unique()

    for i in range(len(values)):
        data[column] = data[column].replace(values[i], i)

    return data

def one_hot_encoding(data, column):

    encoded = pd.get_dummies(data[column], prefix=column, dtype=int)

    data = data.drop(column, axis=1)

    data = pd.concat([data, encoded], axis=1)

    return data

categorical_columns = df.select_dtypes(include="object").columns

print("Categorical Columns:")
print(categorical_columns)
label_data = df.copy()

for column in categorical_columns:
    label_data = label_encoding(label_data, column)

print("\nLabel Encoded Data:")
print(label_data.head())

onehot_data = df.copy()

for column in categorical_columns:
    onehot_data = one_hot_encoding(onehot_data, column)

print("\nOne-Hot Encoded Data:")
print(onehot_data.head())