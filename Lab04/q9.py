import numpy as np
import pandas as pd

def calculate_mean(data):

    return np.sum(data) / len(data)


def calculate_standard_deviation(data):

    mean = calculate_mean(data)

    variance = (
        np.sum((data - mean) ** 2)
        / len(data)
    )

    return np.sqrt(variance)


file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

numeric_df = df.select_dtypes(
    include=np.number
).dropna()

X = numeric_df.values

own_mean = []

own_std = []

for i in range(X.shape[1]):

    column = X[:, i]

    own_mean.append(
        calculate_mean(column)
    )

    own_std.append(
        calculate_standard_deviation(column)
    )

numpy_mean = np.mean(
    X,
    axis=0
)

numpy_std = np.std(
    X,
    axis=0
)


print("Feature\t\tMy Mean\t\tNumPy Mean")

for i, column in enumerate(
    numeric_df.columns
):

    print(
        column,
        "\t",
        own_mean[i],
        "\t",
        numpy_mean[i]
    )


print("\n\nFeature\t\tMy Std\t\tNumPy Std")

for i, column in enumerate(
    numeric_df.columns
):

    print(
        column,
        "\t",
        own_std[i],
        "\t",
        numpy_std[i]
    )