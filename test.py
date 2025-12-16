def test(a, v):
    for i in range(len(a)):
        if a[i] == v:
            return i
        else:
            return -1

print(test([1, 2, 3, 4], 3)) # 2 