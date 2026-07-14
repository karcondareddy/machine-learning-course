def multiply(m1, m2):
    n = len(A)
    res = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += m1[i][k] * m2[k][j]

    return res


def matrixpower(A, m):
    if m == 1:
        return A

    result = A

    for _ in range(2, m + 1):
        result = multiply(result, A)

    return result

n = int(input("Enter order of matrix: "))

A = []
print("Enter matrix:")
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

m = int(input("Enter power: "))

ans = matrixpower(A, m)

print("A^", m, "=")
for row in ans:
    print(row)