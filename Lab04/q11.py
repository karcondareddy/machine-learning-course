import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# GenAI Tool Used: ChatGPT

file_path = "Lab Session Data.xlsx"

data = pd.read_excel(file_path, sheet_name="marketing_campaign")


def minkowski_distance(A, B, p=2):
    return np.sum(np.abs(A - B) ** p) ** (1 / p)


def assign_clusters(data, centroids):
    clusters = []

    for point in data:
        distances = []

        for centroid in centroids:
            distance = minkowski_distance(point, centroid)
            distances.append(distance)

        clusters.append(np.argmin(distances))

    return np.array(clusters)


def calculate_centroids(data, clusters, k):
    centroids = []

    for cluster in range(k):
        points = data[clusters == cluster]

        if len(points) > 0:
            centroid = np.mean(points, axis=0)
        else:
            centroid = data[np.random.randint(len(data))]

        centroids.append(centroid)

    return np.array(centroids)


def k_means(data, k, max_iterations=100):
    random_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[random_indices].copy()

    for _ in range(max_iterations):

        clusters = assign_clusters(data, centroids)

        new_centroids = calculate_centroids(data, clusters, k)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

numeric_data = data.select_dtypes(include=np.number)

numeric_data = numeric_data.dropna()

feature_data = numeric_data.values

k = 3

clusters, centroids = k_means(feature_data, k)

print("K-Means Clustering Results")
print("-" * 40)

for cluster in range(k):
    count = np.sum(clusters == cluster)
    print(f"Cluster {cluster + 1}: {count} data points")


print("\nCentroids")
print("-" * 40)

for cluster in range(k):
    print(f"\nCluster {cluster + 1}:")
    
    for feature, value in zip(numeric_data.columns, centroids[cluster]):
        print(f"{feature:25} : {value:.4f}")


plt.figure(figsize=(10, 6))

for cluster in range(k):
    points = feature_data[clusters == cluster]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster + 1}"
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel(numeric_data.columns[0])
plt.ylabel(numeric_data.columns[1])
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)

plt.show()