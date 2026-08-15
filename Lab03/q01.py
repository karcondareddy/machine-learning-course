import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)


def identify_type(column):
    if column.dtype == "object":
        return "Nominal"
    elif pd.api.types.is_datetime64_any_dtype(column):
        return "Interval"
    elif pd.api.types.is_numeric_dtype(column):
        return "Ratio"
    else:
        return "Unknown"


print(df.head())

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMeasurement Types:")

for column in df.columns:
    print(column, "->", identify_type(df[column]))