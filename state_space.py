import math


def modulus(m, n):
    mod = []
    for j in range(1, m):
        if j == 1:
            mod.append(n % m)
            a_1 = mod[0]
            return a_1
        elif (j > 1) and (n - m + mod[j - 1] > m):
            mod.append(n - m + mod[j - 1] % m)
            return mod
        elif (j > 2) and (n - m + mod[j - 1]) < m:
            mod.append(m - ((n - mod[j - 2]) % m))
            return mod



# def a_i(m, n):
#     a_list = []
#     for i in range(1, m):
#         a_list.append(modulus(m, n, i))
#     return a_list


def x_i(m, n):
    x = 0
    for j in range(0, m - 1):
        if (n - m + modulus(m, n)[j]) % m != 0:
            x += math.floor((n - m + modulus(m, n)[j]))
            return x


def number_of_bounces(m, n):
    N = 2 * (m - 1) + x_i(m, n) + math.floor(n/m) - 1
    return N


print(modulus(6, 8))




