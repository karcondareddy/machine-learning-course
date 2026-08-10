import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_mean(data):

    return np.sum(data) / len(data)


def calculate_variance(data):

    mean = calculate_mean(data)

    return np.sum(
        (data - mean) ** 2
    ) / len(data)


file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

numeric_df = df.select_dtypes(
    include=np.number
).dropna()

feature = numeric_df.columns[0]

data = numeric_df[feature].values

counts, bins = np.histogram(
    data,
    bins=10
)

print("Feature:", feature)

print("\nHistogram Counts:")
print(counts)

print("\nBin Edges:")
print(bins)

print(
    "\nMean:",
    calculate_mean(data)
)

print(
    "Variance:",
    calculate_variance(data)
)

plt.hist(
    data,
    bins=10,
    edgecolor="black"
)

plt.xlabel(feature)
plt.ylabel("Frequency")

plt.title(
    "Histogram of " + feature
)

plt.show()