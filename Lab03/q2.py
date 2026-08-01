import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

# Label Encoding
def my_label_encoder(series):
    unique_values = sorted(series.dropna().unique())

    mapping = {}
    for i, value in enumerate(unique_values):
        mapping[value] = i

    encoded = series.map(mapping)

    return encoded, mapping


# One Hot Encoding
def my_one_hot_encoder(series):

    unique_values = sorted(series.dropna().unique())

    encoded_df = pd.DataFrame()

    for value in unique_values:
        encoded_df[f"{series.name}_{value}"] = (series == value).astype(int)

    return encoded_df