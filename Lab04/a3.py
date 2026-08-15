import pandas as pd
import numpy as np
import time

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


# --------------------------------------------------
# K-Means Version 1: Own Implementation
# --------------------------------------------------

def minkowski_distance_own(A, B, p=2):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)


def assign_clusters_own(data, centroids):
    clusters = []

    for point in data:
        distances = []

        for centroid in centroids:
            distance = minkowski_distance_own(point, centroid)
            distances.append(distance)

        clusters.append(np.argmin(distances))

    return np.array(clusters)


def calculate_centroids_own(data, clusters, k):
    centroids = []

    for cluster in range(k):
        points = data[clusters == cluster]

        if len(points) > 0:
            centroid = np.mean(points, axis=0)
        else:
            centroid = data[np.random.randint(len(data))]

        centroids.append(centroid)

    return np.array(centroids)


def k_means_own(data, k, max_iterations=100):

    random_indices = np.random.choice(
        len(data), k, replace=False
    )

    centroids = data[random_indices].copy()

    for _ in range(max_iterations):

        clusters = assign_clusters_own(
            data, centroids
        )

        new_centroids = calculate_centroids_own(
            data, clusters, k
        )

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# --------------------------------------------------
# K-Means Version 2: AI Implementation
# --------------------------------------------------

def k_means_ai(data, k, max_iterations=100):

    centroids = data[
        np.random.choice(len(data), k, replace=False)
    ].copy()

    for _ in range(max_iterations):

        distances = np.zeros((len(data), k))

        for i in range(k):
            distances[:, i] = np.sqrt(
                np.sum(
                    (data - centroids[i]) ** 2,
                    axis=1
                )
            )

        clusters = np.argmin(distances, axis=1)

        new_centroids = np.array([
            data[clusters == i].mean(axis=0)
            if np.any(clusters == i)
            else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids



numeric_data = data.select_dtypes(include=np.number).dropna()
feature_data = numeric_data.values

k = 3

start_time = time.perf_counter()
clusters_own, centroids_own = k_means_own(feature_data, k)
own_time = time.perf_counter() - start_time


start_time = time.perf_counter()
clusters_ai, centroids_ai = k_means_ai(feature_data, k)
ai_time = time.perf_counter() - start_time


print("K-Means Performance Comparison")
print("-" * 50)

print(f"Own K-Means Time : {own_time:.6f} seconds")
print(f"AI K-Means Time  : {ai_time:.6f} seconds")

print("-" * 50)

if own_time < ai_time:
    print("Own implementation is faster.")

elif ai_time < own_time:
    print("AI implementation is faster.")

else:
    print("Both implementations have similar execution time.")