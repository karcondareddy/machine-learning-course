import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

def one_hot_encoding(data, columns):

    data = pd.get_dummies(
        data,
        columns=columns,
        dtype=int
    )

    return data

categorical_columns = df.select_dtypes(include="object").columns
print("Categorical Columns:")
print(categorical_columns)


print("\nOriginal Dataset:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


encoded_data = one_hot_encoding(df.copy(),categorical_columns)

print("\nEncoded Dataset:")
print("Rows:", encoded_data.shape[0])
print("Columns:", encoded_data.shape[1])

print("\nFeature Dimensionality:")
print("Before Encoding:", df.shape[1])
print("After Encoding:", encoded_data.shape[1])

print("\nEncoded Data:")
print(encoded_data.head())