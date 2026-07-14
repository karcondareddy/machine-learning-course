def pairSum(arr):
    result = []
    count = 0
    n = len(arr)
    tar = 10
    arr.sort()
    left = 0
    right = n-1
    while (left < right):
        s = arr[left] + arr[right]
        if (s == tar):
            result.append((arr[left],arr[right]))
            count+=1
            left+=1
            right-=1
        elif s > tar:
            right-=1
        else :
            left+=1
        
    return result,count

list1 = [2,7,4,1,3,6]
pairs,paircount = pairSum(list1)
print(pairs)
print(paircount)