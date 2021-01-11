a = 0
b = 1

for i in range(1, 50):
    a, b = b, a+b
    print(b)

