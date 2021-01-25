# 题目：求一个3*3矩阵主对角线元素之和。
#
# 程序分析：利用双重for循环控制输入二维数组，再将a[i][i]累加后输出。
import random

ls =[]

for i in range(3):
    ls.append([])
    for j in range(3):
        ls[i].append(random.randint(1,100))
print(ls)

sum =0
for i in range(3):
    sum+=ls[i][i]
print(sum)