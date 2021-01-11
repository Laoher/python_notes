# # 冒泡排序(bubble sort)
# 排序算法参考
# https://www.cnblogs.com/onepixel/articles/7674659.html
#
# 非线性时间比较类排序：通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O(nlogn)，因此称为非线性时间比较类排序。
# 线性时间非比较类排序：不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此称为线性时间非比较类排序。
import random
ls=[]
for i in range(10):
    ls.append(random.randint(1,100))
print(ls)
# 冒泡排序(bubble sort)
for i in range(len(ls)-1,0,-1):
    for j in range(i):
        if ls[j]>ls[j+1]:
            ls[j],ls[j+1] = ls[j+1],ls[j]
#输出
print(ls)

