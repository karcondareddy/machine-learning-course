import pandas as pd

def one_hot_encoding(data, column):

    encoded_data = data.copy()

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

column = "Education"

encoded_df = one_hot_encoding(
    df,
    column
)

print("Original Dataset:")
print(df.head())

print("\nOne-Hot Encoded Dataset:")
print(encoded_df.head())