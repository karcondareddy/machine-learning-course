import pandas as pd

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nPandas Data Types:")
print(df.dtypes)

def identify_measurement_type(series):
    if series.dtype == "object":
        return "Nominal"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "Interval"
    elif pd.api.types.is_integer_dtype(series):
        return "Ratio"
    elif pd.api.types.is_float_dtype(series):
        return "Ratio"
    else:
        return "Unknown"

print("\nMeasurement Types:")
for col in df.columns:
    print(f"{col:25} -> {identify_measurement_type(df[col])}")