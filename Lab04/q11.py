import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(A, B):

    return np.sqrt(
        np.sum((A - B) ** 2)
    )


def assign_clusters(data, centroids):

    labels = []

    for point in data:

        distances = []

        for centroid in centroids:

            distance = euclidean_distance(
                point,
                centroid
            )

            distances.append(distance)

        nearest_cluster = np.argmin(
            distances
        )

        labels.append(
            nearest_cluster
        )

    return np.array(labels)


def update_centroids(data, labels, k):

    centroids = []

    for cluster in range(k):

        cluster_points = data[
            labels == cluster
        ]

        if len(cluster_points) > 0:

            centroid = np.mean(
                cluster_points,
                axis=0
            )

        else:

            centroid = data[
                np.random.randint(
                    0,
                    len(data)
                )
            ]

        centroids.append(
            centroid
        )

    return np.array(centroids)


def k_means(
    data,
    k,
    max_iterations=100
):

    indices = np.random.choice(
        len(data),
        k,
        replace=False
    )

    centroids = data[
        indices
    ].copy()


    for iteration in range(
        max_iterations
    ):

        labels = assign_clusters(
            data,
            centroids
        )

        new_centroids = update_centroids(
            data,
            labels,
            k
        )

        if np.allclose(
            centroids,
            new_centroids
        ):

            break


        centroids = new_centroids


    return labels, centroids

file = "Lab Session Data.xlsx"
sheet = "marketing_campaign"

df = pd.read_excel(file, sheet_name=sheet)


numeric_df = df.select_dtypes(
    include=np.number
).dropna()

data = numeric_df.values

mean = np.mean(
    data,
    axis=0
)

std = np.std(
    data,
    axis=0
)

std[std == 0] = 1

data = (
    data - mean
) / std

k = 3

labels, centroids = k_means(
    data,
    k
)

print("Cluster Labels:")
print(labels)

print("\nCentroids:")
print(centroids)

plt.scatter(
    data[:, 0],
    data[:, 1],
    c=labels
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200
)

plt.xlabel(
    numeric_df.columns[0]
)

plt.ylabel(
    numeric_df.columns[1]
)

plt.title(
    "K-Means Clustering"
)

plt.show()