import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

price = df["Price"].to_numpy()

numpy_mean = np.mean(price)
numpy_variance = np.var(price)

print("NumPy Mean :", numpy_mean)
print("NumPy Variance :", numpy_variance)

def my_mean(arr):
    total = 0
    for value in arr:
        total += value
    return total / len(arr)

def my_variance(arr):
    mean = my_mean(arr)

    total = 0
    for value in arr:
        total += (value - mean) ** 2

    return total / len(arr)

print("\nUser Mean :", my_mean(price))
print("User Variance :", my_variance(price))

numpy_times = []
user_times = []

for i in range(10):

    start = time.perf_counter()
    np.mean(price)
    np.var(price)
    end = time.perf_counter()

    numpy_times.append(end - start)

for i in range(10):

    start = time.perf_counter()
    my_mean(price)
    my_variance(price)
    end = time.perf_counter()

    user_times.append(end - start)

print("\nAverage NumPy Time :", np.mean(numpy_times))
print("Average User Function Time :", np.mean(user_times))


wednesday_price = df[df["Day"] == "Wed"]["Price"]
print(wednesday_price)

print("\nWednesday Mean :", np.mean(wednesday_price))
print("Population Mean :", numpy_mean)

april_price = df[df["Month"] == "Apr"]["Price"]
print("\nApril Mean :", np.mean(april_price))
print("Population Mean :", numpy_mean)

loss = df["Chg%"].apply(lambda x: x < 0)
print(loss)
prob_loss = loss.sum() / len(df)
print("\nProbability of Loss :", prob_loss)

wednesday = df[df["Day"] == "Wed"]
profit_wednesday = wednesday["Chg%"] > 0
prob_profit_wednesday = profit_wednesday.sum() / len(df)
print("Probability of Profit on Wednesday :", prob_profit_wednesday)

conditional_probability = profit_wednesday.sum() / len(wednesday)
print("Conditional Probability (Profit | Wednesday) :", conditional_probability)

plt.figure(figsize=(10,5))
plt.scatter(df["Day"], df["Chg%"])
plt.xlabel("Day of Week")
plt.ylabel("Change %")
plt.title("Change % vs Day")
plt.grid(True)
plt.show()