import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

data = df.select_dtypes(include=np.number)

print("Available Features:")
print(data.columns.tolist())

feature = data.columns[0]

values = data[feature]

print("\nSelected Feature:", feature)

hist, bins = np.histogram(values, bins=10)

print("\nHistogram Counts:")
print(hist)

print("\nBin Edges:")
print(bins)

plt.figure(figsize=(8,5))

plt.hist(values, bins=10, edgecolor="black")

plt.title(f"Histogram of {feature}")

plt.xlabel(feature)

plt.ylabel("Frequency")

plt.grid(True)

plt.show()

print("Mean     :", np.mean(values))
print("Variance :", np.var(values))