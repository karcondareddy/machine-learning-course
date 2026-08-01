import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]

data = df.select_dtypes(include=np.number).values

def euclidean_distance(a, b):
    total = 0

    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2

    return total ** 0.5


# ---------- K-Means ----------
def my_kmeans(data, k, max_iterations=100):

    centroids = data[:k].copy()

    for _ in range(max_iterations):

        clusters = []

        # Assign each point
        for point in data:

            distances = []

            for centroid in centroids:
                distances.append(euclidean_distance(point, centroid))

            clusters.append(np.argmin(distances))

        clusters = np.array(clusters)

        new_centroids = []

        # Update centroids
        for i in range(k):

            cluster_points = data[clusters == i]

            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# Run K-Means
k = 3

clusters, centroids = my_kmeans(data, k)

print("Cluster Labels:")
print(clusters)

print("\nFinal Centroids:")
print(centroids)