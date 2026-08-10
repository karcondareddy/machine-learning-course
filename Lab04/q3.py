import pandas as pd

def one_hot_encode_all(data, categorical_columns):

    encoded_data = data.copy()

    for column in categorical_columns:

        categories = encoded_data[column].dropna().unique()

        for category in categories:

            new_column = (
                column + "_" + str(category)
            )

            encoded_data[new_column] = (
                encoded_data[column] == category
            ).astype(int)

        encoded_data.drop(
            columns=[column],
            inplace=True
        )

    return encoded_data


file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("Categorical Columns:")
print(categorical_columns)

encoded_df = one_hot_encode_all(
    df,
    categorical_columns
)

print("\nOriginal Shape:")
print(df.shape)

print("\nEncoded Shape:")
print(encoded_df.shape)

print(
    "\nOriginal Number of Features:",
    df.shape[1]
)

print(
    "Encoded Number of Features:",
    encoded_df.shape[1]
)