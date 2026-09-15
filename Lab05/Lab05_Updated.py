import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------
# 1. Encoding
# ---------------------------------------------------------

# def encode_data(data):
#     data = data.copy()

#     for column in data.columns:
#         if data[column].dtype == 'object':
#             encoder = LabelEncoder()
#             data[column] = encoder.fit_transform(data[column].astype(str))

#     return data


def encode_data(data, column):
    values = data[column].unique()
    encoding = {value: index for index, value in enumerate(values)}
    data[column] = data[column].map(encoding)

    return data


# ---------------------------------------------------------
# 2. Data Imputation
# ---------------------------------------------------------

def impute_data(data):
    data = data.copy()

    for column in data.columns:
        if data[column].isnull().sum() > 0:

            if data[column].dtype == 'object':
                data[column] = data[column].fillna(data[column].mode()[0])
            else:
                data[column] = data[column].fillna(data[column].median())

    return data


# ---------------------------------------------------------
# 3. Distance Calculation
# ---------------------------------------------------------

def minkowski_distance(A, B, p=2):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)


# ---------------------------------------------------------
# 4. Sorting Algorithms
# ---------------------------------------------------------

def fast_sort(distances):
    return sorted(distances, key=lambda x: x[0])

def bubble_sort(distances):
    arr = distances.copy()
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):

            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


def selection_sort(distances):
    arr = distances.copy()
    n = len(arr)

    for i in range(n):

        min_index = i

        for j in range(i + 1, n):

            if arr[j][0] < arr[min_index][0]:
                min_index = j

        arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr


def insertion_sort(distances):
    arr = distances.copy()

    for i in range(1, len(arr)):

        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j][0] > key[0]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


# ---------------------------------------------------------
# 5. Sorting Selection
# ---------------------------------------------------------

def sort_distances(distances, method="fast"):
    if method == "fast":
        return fast_sort(distances)

    elif method == "bubble":
        return bubble_sort(distances)

    elif method == "selection":
        return selection_sort(distances)

    elif method == "insertion":
        return insertion_sort(distances)

    else:
        raise ValueError("Invalid sorting method")


# ---------------------------------------------------------
# 6. Find k Nearest Neighbors
# ---------------------------------------------------------

def identify_neighbors(X_train, y_train, test_point, k, distance_p=2, sorting_method="fast"):

    distances = []

    for i in range(len(X_train)):

        distance = minkowski_distance(X_train[i], test_point, distance_p)

        distances.append((distance, y_train[i], i))

    distances = sort_distances(distances, sorting_method)

    return distances[:k]


# ---------------------------------------------------------
# 7. Majority Voting
# ---------------------------------------------------------

def majority_vote(neighbors):

    votes = {}

    for distance, label, index in neighbors:

        if label not in votes:
            votes[label] = 0

        votes[label] += 1

    # Tie breaking:
    # choose the class having the nearest neighbor
    max_votes = max(votes.values())

    candidates = [
        label for label, count in votes.items()
        if count == max_votes
    ]

    if len(candidates) == 1:
        return candidates[0]

    for distance, label, index in neighbors:

        if label in candidates:
            return label


# ---------------------------------------------------------
# 8. Fit Function
# ---------------------------------------------------------

def fit(X_train, y_train):

    model = {
        "X_train": np.array(X_train),
        "y_train": np.array(y_train)
    }

    return model


# ---------------------------------------------------------
# 9. Predict Function
# ---------------------------------------------------------

def predict(model, X_test, k=3, distance_p=2, sorting_method="fast"):

    X_train = model["X_train"]
    y_train = model["y_train"]

    predictions = []

    for test_point in X_test:

        neighbors = identify_neighbors( X_train, y_train, test_point, k, distance_p, sorting_method)

        prediction = majority_vote(neighbors)

        predictions.append(prediction)

    return np.array(predictions)


# ---------------------------------------------------------
# 10. Score Function
# ---------------------------------------------------------

def score(model, X_test, y_test, k=3, distance_p=2, sorting_method="fast"):

    predictions = predict( model, X_test, k, distance_p, sorting_method)

    accuracy = np.mean(predictions == np.array(y_test))

    return accuracy


# ---------------------------------------------------------
# 11. Weighted k-NN
# ---------------------------------------------------------

def weighted_vote(neighbors):

    votes = {}

    for distance, label, index in neighbors:

        # Avoid division by zero
        weight = 1 / (distance + 1e-10)

        if label not in votes:
            votes[label] = 0

        votes[label] += weight

    max_weight = max(votes.values())

    candidates = [
        label for label, weight in votes.items()
        if weight == max_weight
    ]

    if len(candidates) == 1:
        return candidates[0]

    # Tie breaking
    for distance, label, index in neighbors:

        if label in candidates:
            return label


# ---------------------------------------------------------
# 12. Weighted Predict
# ---------------------------------------------------------

def weighted_predict(model, X_test, k=3, distance_p=2, sorting_method="fast"):

    X_train = model["X_train"]
    y_train = model["y_train"]

    predictions = []

    for test_point in X_test:

        neighbors = identify_neighbors(X_train, y_train, test_point, k, distance_p, sorting_method)

        prediction = weighted_vote(neighbors)

        predictions.append(prediction)

    return np.array(predictions)


# ---------------------------------------------------------
# 13. Weighted Score
# ---------------------------------------------------------

def weighted_score(model, X_test, y_test, k=3, distance_p=2, sorting_method="fast"):

    predictions = weighted_predict( model, X_test, k, distance_p, sorting_method)

    accuracy = np.mean(predictions == np.array(y_test))

    return accuracy


# =========================================================
# MAIN PROGRAM
# =========================================================

# Read dataset
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")


# Remove rows having missing target value
data = data.dropna(subset=["Response"])


# Select numerical features
features = [
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Response"
]

data = data[features]


# Imputation
data = impute_data(data)


# Separate X and y
X = data.drop("Response", axis=1).values
y = data["Response"].values


# ---------------------------------------------------------
# A3: Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ---------------------------------------------------------
# Our k-NN model
# ---------------------------------------------------------

my_model = fit(X_train, y_train)


# ---------------------------------------------------------
# A4-A6: k = 3
# ---------------------------------------------------------

my_accuracy = score(my_model, X_test, y_test, k=3, sorting_method="fast")

my_predictions = predict(my_model, X_test, k=3, sorting_method="fast")


print("Our kNN Accuracy:", my_accuracy)
print("Our kNN Predictions:", my_predictions)


# ---------------------------------------------------------
# sklearn kNN
# ---------------------------------------------------------

sklearn_model = KNeighborsClassifier(n_neighbors=3)

sklearn_model.fit(X_train, y_train)

sklearn_accuracy = sklearn_model.score(X_test, y_test)

sklearn_predictions = sklearn_model.predict(X_test)

print("Sklearn kNN Accuracy:", sklearn_accuracy)
print("Sklearn Predictions:", sklearn_predictions)


# ---------------------------------------------------------
# A8: Compare for different k values
# ---------------------------------------------------------

k_values = range(1, 16)

my_accuracies = []
sklearn_accuracies = []
weighted_accuracies = []


for k in k_values:

    # Our normal kNN
    acc = score( my_model, X_test, y_test, k=k, sorting_method="fast")

    my_accuracies.append(acc)


    # sklearn kNN
    model = KNeighborsClassifier( n_neighbors=k)

    model.fit( X_train, y_train)

    acc_sklearn = model.score( X_test, y_test)

    sklearn_accuracies.append(acc_sklearn)


    # Weighted kNN
    weighted_acc = weighted_score( my_model, X_test, y_test, k=k, sorting_method="fast")

    weighted_accuracies.append(weighted_acc)


# ---------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    my_accuracies,
    marker="o",
    label="Our kNN"
)

plt.plot(
    k_values,
    sklearn_accuracies,
    marker="s",
    label="Sklearn kNN"
)

plt.plot(
    k_values,
    weighted_accuracies,
    marker="^",
    label="Weighted kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")

plt.title("kNN Accuracy Comparison")

plt.legend()
plt.grid(True)

plt.show()


# ---------------------------------------------------------
# Print accuracy values
# ---------------------------------------------------------

print("\nk    Our kNN    Sklearn kNN    Weighted kNN")

for i, k in enumerate(k_values):

    print(k, round(my_accuracies[i], 4), round(sklearn_accuracies[i], 4), round(weighted_accuracies[i], 4))
