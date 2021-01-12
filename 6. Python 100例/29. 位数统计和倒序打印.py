# 题目：给一个不多于5位的正整数，要求：一、求它是几位数，二、逆序打印出各位数字。
#
# 程序分析：学会分解出每一位数。
#
# 1-99999
a = int(input("Please input a number not more than 5 digits:\n"))


for i in range(1,6):
    if a//10==0:
        print(int(a))
        print()
        print("number of digits is:",i)
        break
    else:
        digit = int(a % 10)
        print(digit)
        a = (a - digit) / 10



