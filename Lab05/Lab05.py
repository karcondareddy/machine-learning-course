import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# A1.a: Encode categorical data
def encode_categorical(df):
    encoded_df = df.copy()
    for col in encoded_df.select_dtypes(include=['object', 'category']).columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes
    return encoded_df

# A1.b: Impute missing values using mean, median, or mode
def impute_data(df, strategy='median'):
    imputed_df = df.copy()
    for col in imputed_df.columns:
        if imputed_df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(imputed_df[col]):
                val = imputed_df[col].mean()
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(imputed_df[col]):
                val = imputed_df[col].median()
            else:
                val = imputed_df[col].mode()[0]
            imputed_df[col] = imputed_df[col].fillna(val)
    return imputed_df

# A1.c: Calculate distance between points
def calculate_distance(pt1, pt2, metric='euclidean'):
    if metric == 'manhattan':
        return np.sum(np.abs(pt1 - pt2))
    return np.sqrt(np.sum((pt1 - pt2) ** 2))

# A1.d: Sorting algorithms (Bubble, Selection, Insertion)
def custom_sort(distances, method='selection'):
    arr = distances.copy()
    n = len(arr)
    if method == 'bubble':
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j][0] > arr[j + 1][0]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
    elif method == 'selection':
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j][0] < arr[min_idx][0]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    elif method == 'insertion':
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and key[0] < arr[j][0]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    return arr

# A1.e: Identify k-nearest neighbors
def identify_neighbors(x, X_train, y_train, k, sort_method='selection', metric='euclidean'):
    distances = []
    for i, x_train_pt in enumerate(X_train):
        dist = calculate_distance(x, x_train_pt, metric)
        distances.append((dist, i, y_train[i]))
    sorted_distances = custom_sort(distances, method=sort_method)
    return sorted_distances[:k]

# A1.f & A2: Assign class label using majority voting (supports weighted voting)
def assign_class_label(k_neighbors, weights='uniform'):
    votes = {}
    for dist, _, label in k_neighbors:
        weight = 1.0 if weights == 'uniform' else 1.0 / (dist + 1e-5)
        votes[label] = votes.get(label, 0.0) + weight
    return max(votes, key=votes.get)

# A7.a: Fit function
def custom_knn_fit(X, y):
    return np.array(X), np.array(y)

# A7.b: Predict function
def custom_knn_predict(X_test, X_train, y_train, k=3, weights='uniform', sort_method='selection', metric='euclidean'):
    X_test_array = np.array(X_test)
    predictions = []
    for x in X_test_array:
        neighbors = identify_neighbors(x, X_train, y_train, k, sort_method, metric)
        label = assign_class_label(neighbors, weights)
        predictions.append(label)
    return np.array(predictions)

# A7.c: Score function
def custom_knn_score(X_test, y_test, X_train, y_train, k=3, weights='uniform', sort_method='selection', metric='euclidean'):
    predictions = custom_knn_predict(X_test, X_train, y_train, k, weights, sort_method, metric)
    return np.mean(predictions == np.array(y_test))

# ==========================================
# Main Execution Steps
# ==========================================

# Load dataset
df = pd.read_csv("diabetic_data.csv", na_values=['?', 'None'])

# Impute and encode data
df = impute_data(df, strategy='median')
df = encode_categorical(df)

# Filter to 2 classes for binary classification (A3 requirement)
target_col = 'readmitted' if 'readmitted' in df.columns else df.columns[-1]
classes = df[target_col].unique()
if len(classes) > 2:
    df = df[df[target_col].isin(classes[:2])]

# Sample data to make execution of custom sorting faster
df = df.sample(n=500, random_state=42)
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split (A3)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.3, random_state=42)

# Scikit-learn kNN implementation (A4, A5, A6)
sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train_raw, y_train_raw)
sklearn_preds = sklearn_knn.predict(X_test_raw)
sklearn_acc = sklearn_knn.score(X_test_raw, y_test_raw)
print(f"Sklearn kNN (k=3) Accuracy: {sklearn_acc:.4f}")

# Prepare data for custom functions (A7.a)
X_train, y_train = custom_knn_fit(X_train_raw, y_train_raw)
X_test, y_test = np.array(X_test_raw), np.array(y_test_raw)

# Comparative Analysis loop (A8, A9)
k_values = [1, 3, 5, 7, 9]
acc_sklearn = []
acc_custom_uniform = []
acc_custom_weighted = []

for k in k_values:
    # Sklearn (A8)
    sk_model = KNeighborsClassifier(n_neighbors=k)
    sk_model.fit(X_train_raw, y_train_raw)
    acc_sklearn.append(sk_model.score(X_test_raw, y_test_raw))

    # Custom Uniform (A8)
    acc_uni = custom_knn_score(X_test, y_test, X_train, y_train, k=k, weights='uniform')
    acc_custom_uniform.append(acc_uni)

    # Custom Weighted (A9)
    acc_wht = custom_knn_score(X_test, y_test, X_train, y_train, k=k, weights='distance')
    acc_custom_weighted.append(acc_wht)
    
    print(f"Completed evaluation for k={k}")

# Plotting results (A8, A9)
plt.figure(figsize=(8, 5))
plt.plot(k_values, acc_sklearn, label='Sklearn Uniform', marker='o')
plt.plot(k_values, acc_custom_uniform, label='Custom Uniform', marker='s')
plt.plot(k_values, acc_custom_weighted, label='Custom Weighted', marker='^')
plt.title('k-NN Accuracy Comparison')
plt.xlabel('k Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()