import math
v = []


def residues(m, n):
    h = [i for i in range(n - m + 1, n + 1)]
    for j in h:
        if j % m != 0:
            v.append((math.floor(j / m)) + 2)
        else:
            v.append((j / m) - 1)
    return v

a = residues(8, 13)
b = sum(a)
print(b)

# def validation(g, w):
#
#         for y in range(1, g + 1):
#
#              print(residues(g, w))
#
#
# validation(5, 10)



