def highestChar(s):
    freq = {}

    for ch in s.lower():
        if ch.isalpha():
            if ch in freq:
                freq[ch] += 1
            else:
                freq[ch] = 1

    maxChar = ""
    maxCount = 0

    for ch in freq:
        if freq[ch] > maxCount:
            maxCount = freq[ch]
            maxChar = ch

    return maxChar, maxCount


string = input("Enter a string: ")

char, count = highestChar(string)

print("Highest occurring character:", char)
print("Occurrence count:", count)