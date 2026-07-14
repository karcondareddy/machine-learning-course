import random
nums = []

for i in range(25):
    nums.append(random.randint(1, 10))

print("Random Numbers:")
print(nums)

# Mean
total = 0
for num in nums:
    total += num

mean = total / len(nums)

# Median
nums.sort()
n = len(nums)

if n % 2 == 0:
    median = (nums[n // 2 - 1] + nums[n // 2 - 1 + 1]) / 2
else:
    median = nums[n // 2]

# Mode
freq = {}

for num in nums:
    if num in freq:
        freq[num] += 1
    else:
        freq[num] = 1

mode = nums[0]
maxCount = 0

for key in freq:
    if freq[key] > maxCount:
        maxCount = freq[key]
        mode = key

print("Sorted List:", nums)
print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
print("Mode Count =", maxCount)