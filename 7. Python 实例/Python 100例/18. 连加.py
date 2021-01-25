# 题目：求s=a+aa+aaa+aaaa+aa...a的值，其中a是一个数字。例如2+22+222+2222+22222(此时共有5个数相加)，几个数相加由键盘控制。
# 程序分析：关键是计算出每一项的值。

n = int(input("how many number to add:"))
d = int(input("the digit number:"))


def adding(n, d):
    sum = 0
    if n != 0:
        for j in range(n):
            for i in range(j+1):
                sum += d * 10**i
    return sum

print(adding(n,d))