import pandas as pd
import numpy as np

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)


def distance(A, B):
    total = 0

    for i in range(len(A)):
        total = total + (A[i] - B[i]) ** 2

    return total ** 0.5


def assign_clusters(data, centroids):
    clusters = []

    for point in data:
        distances = []

        for centroid in centroids:
            d = distance(point, centroid)
            distances.append(d)

        clusters.append(np.argmin(distances))

    return np.array(clusters)


def find_centroids(data, clusters, k):
    centroids = []

    for i in range(k):
        points = data[clusters == i]
        centroid = points.mean(axis=0)
        centroids.append(centroid)

    return np.array(centroids)


def kmeans(data, k):
    centroids = data[:k].copy()

    for i in range(100):

        clusters = assign_clusters(data, centroids)

        new_centroids = find_centroids(data, clusters, k)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

data = df.select_dtypes(include="number").dropna().values

k = 3
clusters, centroids = kmeans(data, k)

print("K-Means Clustering Results")

for i in range(k):
    print("Cluster", i + 1, ":", sum(clusters == i), "data points")


print("\nCentroids:")

for i in range(k):
    print("Cluster", i + 1)

    for value in centroids[i]:
        print("%.2f" % value, end=" ")

    print()