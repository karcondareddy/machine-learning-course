def ranges(arr):
    n = len(arr)
    if n <3:
        return "Range is not possible"

    mins = arr[0]
    maxs = arr[0]

    for i in range(n):
        if arr[i] < mins:
            mins = arr[i]
        if arr[i] > maxs:
            maxs = arr[i]
    
    ran = maxs - mins

    return ran,mins,maxs

arr = list(map(int, input("Enter elements: ").split()))
rangess,minss,maxss = ranges(arr)
print("Range is:",rangess,"(",maxss,", ",minss,")")