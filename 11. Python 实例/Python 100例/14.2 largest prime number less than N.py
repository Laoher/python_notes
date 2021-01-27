N = int(input("请输入数字"))
if N < 3 or N % 1 != 0:
    print("The number doesn't fullfill the requirement, please input again")
    N = input("请输入数字")

print(N)

if N == 3:
    print("the prime number is 2")
else:
    # 假设N是100，n就是99
    n = 0
    for j in range(1, N - 2):
        n = N - j
        print(n)
        number = 0
        for i in range(2, n):
            if n % i == 0:
                number = i
                break
        if number == n - 1:
            print(number)
            break

    print(n)



# print("n is not prime number")
