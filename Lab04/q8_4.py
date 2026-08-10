import numpy as np
import pandas as pd

def calculate_mean(data):

    return np.sum(data) / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    return np.sum(
        (data - mean) ** 2
    ) / len(data)


def calculate_standard_deviation(data):

    return np.sqrt(
        calculate_variance(data)
    )


def calculate_dataset_statistics(data):

    numerical_data = data.select_dtypes(
        include=np.number
    )

    for column in numerical_data.columns:

        values = numerical_data[
            column
        ].dropna().values

        mean = calculate_mean(values)

        variance = calculate_variance(values)

        standard_deviation = (
            calculate_standard_deviation(values)
        )

        print("\nFeature:", column)
        print("Mean:", mean)
        print("Variance:", variance)
        print(
            "Standard Deviation:",
            standard_deviation
        )


file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

calculate_dataset_statistics(df)