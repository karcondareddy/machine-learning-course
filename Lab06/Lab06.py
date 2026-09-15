import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import unittest

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
# 1. DATA IMPUTATION
# =========================================================

# GenAI tool used: ChatGPT

def impute_data(data):
    data = data.copy()
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna(data[column].mode()[0])
            else:
                data[column] = data[column].fillna(data[column].median())
    return data


# =========================================================
# 2. ENCODING
# =========================================================

# GenAI tool used: ChatGPT

def encode_data(data, column):
    data = data.copy()
    values = data[column].unique()
    encoding = {value: index for index, value in enumerate(values)}
    data[column] = data[column].map(encoding)
    return data


# =========================================================
# 3. MINKOWSKI DISTANCE
# =========================================================

# GenAI tool used: ChatGPT

def minkowski_distance(A, B, p=2):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)


# =========================================================
# 4. FAST SORTING
# =========================================================

# GenAI tool used: ChatGPT
# Python sorted() uses Timsort and is much faster than Bubble Sort.

def fast_sort(distances):
    return sorted(distances, key=lambda x: x[0])


# =========================================================
# 5. BUBBLE SORT
# =========================================================

def bubble_sort(distances):
    arr = distances.copy()
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr


# =========================================================
# 6. SELECTION SORT
# =========================================================

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


# =========================================================
# 7. INSERTION SORT
# =========================================================

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


# =========================================================
# 8. SORTING SELECTION
# =========================================================

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


# =========================================================
# 9. FIND K NEAREST NEIGHBORS
# =========================================================

# GenAI tool used: ChatGPT

def identify_neighbors(X_train, y_train, test_point, k, distance_p=2, sorting_method="fast"):
    distances = []

    for i in range(len(X_train)):
        distance = minkowski_distance(X_train[i], test_point, distance_p)
        distances.append((distance, y_train[i], i))

    distances = sort_distances(distances, sorting_method)
    return distances[:k]


# =========================================================
# 10. MAJORITY VOTING
# =========================================================

# GenAI tool used: ChatGPT

def majority_vote(neighbors):
    votes = {}

    for distance, label, index in neighbors:
        if label not in votes:
            votes[label] = 0
        votes[label] += 1

    max_votes = max(votes.values())
    candidates = [label for label, count in votes.items() if count == max_votes]

    if len(candidates) == 1:
        return candidates[0]

    for distance, label, index in neighbors:
        if label in candidates:
            return label


# =========================================================
# 11. FIT
# =========================================================

# GenAI tool used: ChatGPT

def fit(X_train, y_train):
    model = {"X_train": np.array(X_train), "y_train": np.array(y_train)}
    return model


# =========================================================
# 12. PREDICT
# =========================================================

# GenAI tool used: ChatGPT

def predict(model, X_test, k=3, distance_p=2, sorting_method="fast"):
    X_train = model["X_train"]
    y_train = model["y_train"]
    predictions = []

    for test_point in X_test:
        neighbors = identify_neighbors(X_train, y_train, test_point, k, distance_p, sorting_method)
        prediction = majority_vote(neighbors)
        predictions.append(prediction)

    return np.array(predictions)


# =========================================================
# 13. SCORE
# =========================================================

# GenAI tool used: ChatGPT

def score(model, X_test, y_test, k=3, distance_p=2, sorting_method="fast"):
    predictions = predict(model, X_test, k, distance_p, sorting_method)
    return accuracy_score(y_test, predictions)


# =========================================================
# 14. WEIGHTED VOTE
# =========================================================

# GenAI tool used: ChatGPT

def weighted_vote(neighbors):
    votes = {}

    for distance, label, index in neighbors:
        weight = 1 / (distance + 1e-10)

        if label not in votes:
            votes[label] = 0

        votes[label] += weight

    max_weight = max(votes.values())
    candidates = [label for label, weight in votes.items() if weight == max_weight]

    if len(candidates) == 1:
        return candidates[0]

    for distance, label, index in neighbors:
        if label in candidates:
            return label


# =========================================================
# 15. WEIGHTED PREDICT
# =========================================================

def weighted_predict(model, X_test, k=3, distance_p=2, sorting_method="fast"):
    X_train = model["X_train"]
    y_train = model["y_train"]
    predictions = []

    for test_point in X_test:
        neighbors = identify_neighbors(X_train, y_train, test_point, k, distance_p, sorting_method)
        prediction = weighted_vote(neighbors)
        predictions.append(prediction)

    return np.array(predictions)


# =========================================================
# 16. WEIGHTED SCORE
# =========================================================

def weighted_score(model, X_test, y_test, k=3, distance_p=2, sorting_method="fast"):
    predictions = weighted_predict(model, X_test, k, distance_p, sorting_method)
    return accuracy_score(y_test, predictions)


# =========================================================
# 17. GENAI k-NN
# =========================================================

# GenAI tool used: ChatGPT
# Vectorized NumPy implementation with np.argpartition()
# for faster nearest-neighbor selection.

def genai_knn_predict(X_train, y_train, X_test, k=3, p=2):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    predictions = []

    for test_point in X_test:
        distances = np.sum(np.abs(X_train - test_point) ** p, axis=1) ** (1 / p)
        nearest_indices = np.argpartition(distances, k - 1)[:k]
        nearest_labels = y_train[nearest_indices]

        values, counts = np.unique(nearest_labels, return_counts=True)
        prediction = values[np.argmax(counts)]
        predictions.append(prediction)

    return np.array(predictions)


# =========================================================
# 18. METRICS
# =========================================================

# GenAI tool used: ChatGPT

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}


# =========================================================
# 19. AVERAGE TIME
# =========================================================

# GenAI tool used: ChatGPT

def average_time(prediction_function, runs=10):
    times = []

    for i in range(runs):
        start_time = time.perf_counter()
        prediction_function()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times)


# =========================================================
# 20. LOAD DATASET
# =========================================================

data = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")


# =========================================================
# 21. REMOVE MISSING TARGET
# =========================================================

data = data.dropna(subset=["Response"])


# =========================================================
# 22. SELECT FEATURES
# =========================================================

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


# =========================================================
# 23. IMPUTATION
# =========================================================

data = impute_data(data)


# =========================================================
# 24. X AND y
# =========================================================

X = data.drop("Response", axis=1).values
y = data["Response"].values


# =========================================================
# 25. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Total samples      :", len(X))
print("Training samples   :", len(X_train))
print("Testing samples    :", len(X_test))
print("Number of features :", X.shape[1])


# =========================================================
# A1 - LAB05 EXPERIMENTS
# =========================================================

my_model = fit(X_train, y_train)


# ---------------------------------------------------------
# Manual k-NN
# ---------------------------------------------------------

my_predictions = predict(my_model, X_test, k=3, sorting_method="fast")
my_accuracy = accuracy_score(y_test, my_predictions)

print("\n========================================")
print("A1 - MANUAL k-NN")
print("========================================")

print("Accuracy:", round(my_accuracy, 4))


# ---------------------------------------------------------
# Scikit-learn k-NN
# ---------------------------------------------------------

sklearn_model = KNeighborsClassifier(n_neighbors=3)
sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print("\n========================================")
print("A1 - SCIKIT-LEARN k-NN")
print("========================================")

print("Accuracy:", round(sklearn_accuracy, 4))


# ---------------------------------------------------------
# Weighted k-NN
# ---------------------------------------------------------

weighted_predictions = weighted_predict(my_model, X_test, k=3, sorting_method="fast")
weighted_accuracy = accuracy_score(y_test, weighted_predictions)

print("\n========================================")
print("A1 - WEIGHTED k-NN")
print("========================================")

print("Accuracy:", round(weighted_accuracy, 4))


# =========================================================
# A1 - DIFFERENT k VALUES
# =========================================================

k_values = range(1, 16)

my_accuracies = []
sklearn_accuracies = []
weighted_accuracies = []


for k in k_values:

    predictions = predict(my_model, X_test, k=k, sorting_method="fast")
    acc = accuracy_score(y_test, predictions)
    my_accuracies.append(acc)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions_sklearn = model.predict(X_test)
    acc_sklearn = accuracy_score(y_test, predictions_sklearn)
    sklearn_accuracies.append(acc_sklearn)

    predictions_weighted = weighted_predict(my_model, X_test, k=k, sorting_method="fast")
    acc_weighted = accuracy_score(y_test, predictions_weighted)
    weighted_accuracies.append(acc_weighted)


print("\n========================================")
print("A1 - k VALUE COMPARISON")
print("========================================")

print("\nk\tManual\t\tSklearn\t\tWeighted")
print("-" * 55)

for i, k in enumerate(k_values):
    print(k, "\t", round(my_accuracies[i], 4), "\t\t", round(sklearn_accuracies[i], 4), "\t\t", round(weighted_accuracies[i], 4))


# =========================================================
# A1 - GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(k_values, my_accuracies, marker="o", label="Manual k-NN")
plt.plot(k_values, sklearn_accuracies, marker="s", label="Scikit-learn k-NN")
plt.plot(k_values, weighted_accuracies, marker="^", label="Weighted k-NN")

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# A3 - MANUAL k-NN
# =========================================================

print("\n========================================")
print("A3 - MANUAL k-NN PERFORMANCE")
print("========================================")

manual_predictions = predict(my_model, X_test, k=3, sorting_method="fast")
manual_metrics = calculate_metrics(y_test, manual_predictions)

manual_time = average_time(
    lambda: predict(my_model, X_test, k=3, sorting_method="fast"),
    runs=10
)

print("Accuracy :", round(manual_metrics["Accuracy"], 4))
print("Precision:", round(manual_metrics["Precision"], 4))
print("Recall   :", round(manual_metrics["Recall"], 4))
print("F1-score :", round(manual_metrics["F1-score"], 4))
print("Average time (10 runs):", round(manual_time, 6), "seconds")


# =========================================================
# A3 - SCIKIT-LEARN k-NN
# =========================================================

print("\n========================================")
print("A3 - SCIKIT-LEARN k-NN PERFORMANCE")
print("========================================")

sklearn_model = KNeighborsClassifier(n_neighbors=3)
sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)
sklearn_metrics = calculate_metrics(y_test, sklearn_predictions)

sklearn_time = average_time(
    lambda: sklearn_model.predict(X_test),
    runs=10
)

print("Accuracy :", round(sklearn_metrics["Accuracy"], 4))
print("Precision:", round(sklearn_metrics["Precision"], 4))
print("Recall   :", round(sklearn_metrics["Recall"], 4))
print("F1-score :", round(sklearn_metrics["F1-score"], 4))
print("Average time (10 runs):", round(sklearn_time, 6), "seconds")


# =========================================================
# A3 - GENAI k-NN
# =========================================================

print("\n========================================")
print("A3 - GENAI k-NN PERFORMANCE")
print("========================================")

genai_predictions = genai_knn_predict(X_train, y_train, X_test, k=3, p=2)
genai_metrics = calculate_metrics(y_test, genai_predictions)

genai_time = average_time(
    lambda: genai_knn_predict(X_train, y_train, X_test, k=3, p=2),
    runs=10
)

print("Accuracy :", round(genai_metrics["Accuracy"], 4))
print("Precision:", round(genai_metrics["Precision"], 4))
print("Recall   :", round(genai_metrics["Recall"], 4))
print("F1-score :", round(genai_metrics["F1-score"], 4))
print("Average time (10 runs):", round(genai_time, 6), "seconds")


# =========================================================
# A3 - FINAL COMPARISON TABLE
# =========================================================

comparison = pd.DataFrame({
    "Method": ["Manual k-NN", "Scikit-learn k-NN", "GenAI k-NN"],
    "Accuracy": [
        manual_metrics["Accuracy"],
        sklearn_metrics["Accuracy"],
        genai_metrics["Accuracy"]
    ],
    "Precision": [
        manual_metrics["Precision"],
        sklearn_metrics["Precision"],
        genai_metrics["Precision"]
    ],
    "Recall": [
        manual_metrics["Recall"],
        sklearn_metrics["Recall"],
        genai_metrics["Recall"]
    ],
    "F1-score": [
        manual_metrics["F1-score"],
        sklearn_metrics["F1-score"],
        genai_metrics["F1-score"]
    ],
    "Average Time (sec)": [
        manual_time,
        sklearn_time,
        genai_time
    ]
})


print("\n========================================")
print("A3 - FINAL PERFORMANCE COMPARISON")
print("========================================")

print(comparison.to_string(index=False))


# =========================================================
# SAVE COMPARISON TABLE
# =========================================================

comparison.to_csv("Lab06_Performance_Comparison.csv", index=False)

print("\nPerformance comparison saved to:")
print("Lab06_Performance_Comparison.csv")


# =========================================================
# A2 - UNIT TEST CASES
# =========================================================

# GenAI tool used: ChatGPT

class TestKNNFunctions(unittest.TestCase):

    def test_minkowski_distance(self):
        A = np.array([0, 0])
        B = np.array([3, 4])
        result = minkowski_distance(A, B, p=2)
        self.assertEqual(result, 5.0)

    def test_fast_sort(self):
        distances = [(5, 1, 0), (2, 0, 1), (8, 1, 2), (1, 0, 3)]
        result = fast_sort(distances)
        result_distances = [item[0] for item in result]
        self.assertEqual(result_distances, [1, 2, 5, 8])

    def test_majority_vote(self):
        neighbors = [(1.0, 1, 0), (2.0, 1, 1), (3.0, 0, 2)]
        result = majority_vote(neighbors)
        self.assertEqual(result, 1)

    def test_weighted_vote(self):
        neighbors = [(1.0, 1, 0), (5.0, 0, 1), (6.0, 0, 2)]
        result = weighted_vote(neighbors)
        self.assertEqual(result, 1)

    def test_fit(self):
        X_train_test = np.array([[1, 2], [3, 4]])
        y_train_test = np.array([0, 1])

        model = fit(X_train_test, y_train_test)

        self.assertEqual(len(model["X_train"]), 2)
        self.assertEqual(len(model["y_train"]), 2)

    def test_prediction_output(self):
        X_train_test = np.array([[0, 0], [1, 1], [10, 10]])
        y_train_test = np.array([0, 0, 1])
        X_test_test = np.array([[0.2, 0.2]])

        model = fit(X_train_test, y_train_test)

        predictions = predict(model, X_test_test, k=1, sorting_method="fast")

        self.assertEqual(len(predictions), 1)

    def test_metrics(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["Precision"], 1.0)
        self.assertEqual(metrics["Recall"], 1.0)
        self.assertEqual(metrics["F1-score"], 1.0)


# =========================================================
# RUN UNIT TESTS
# =========================================================

print("\n========================================")
print("A2 - UNIT TEST RESULTS")
print("========================================")

test_result = unittest.TextTestRunner(verbosity=2).run(
    unittest.TestLoader().loadTestsFromTestCase(TestKNNFunctions)
)


# =========================================================
# TEST SUMMARY
# =========================================================

print("\n========================================")
print("UNIT TEST SUMMARY")
print("========================================")

print("Tests Run :", test_result.testsRun)
print("Failures  :", len(test_result.failures))
print("Errors    :", len(test_result.errors))

if test_result.wasSuccessful():
    print("Result    : ALL TESTS PASSED")
else:
    print("Result    : SOME TESTS FAILED")
