import pandas as pd

def label_encoding(data, column):

    categories = data[column].dropna().unique()

    mapping = {}

    for index, category in enumerate(categories):
        mapping[category] = index

    encoded_data = data.copy()

    encoded_data[column] = (
        encoded_data[column].map(mapping)
    )

    return encoded_data, mapping

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

column = "Education"

encoded_df, mapping = label_encoding(
    df,
    column
)

print("Encoding Mapping:")
print(mapping)

print("\nOriginal values:")
print(df[column].head())

print("\nEncoded values:")
print(encoded_df[column].head())