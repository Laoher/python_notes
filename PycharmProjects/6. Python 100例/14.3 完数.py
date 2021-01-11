# 题目：一个数如果恰好等于它的因子之和，这个数就称为"完数"。例如6=1＋2＋3.编程找出1000以内的所有完数。

from functools import reduce

for n in range(2,1000):
    ls =[]
    for i in range(1,n):
        if n%i==0:
            ls.append(i)
    if reduce(lambda x,y:x+y,ls)==n:
        print(n)