import numpy as np

# GenAI Tool Used: ChatGPT

def minkowski_distance(A, B, p):
    distance = np.sum(np.abs(A - B) ** p) ** (1 / p)

    return distance

A = np.array([1, 2, 3, 4])
B = np.array([5, 6, 7, 8])

for p in [1, 2]:
    distance = minkowski_distance(A, B, p)

    if p == 1:
        print("Manhattan Distance:", distance)
    elif p == 2:
        print("Euclidean Distance:", distance)