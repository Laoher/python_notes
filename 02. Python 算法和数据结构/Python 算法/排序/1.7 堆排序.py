# 堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。
# 堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。

import random
ls = []
for i in range(10):
    ls.append(random.randint(1, 100))

print(ls)
# 堆排序
for j in range(len(ls),1,-1):
    for i in range(j-1,0,-1):
        if ls[i]>ls[int((i+1)/2-1)]:
            ls[i],ls[int((i+1)/2-1)]=ls[int((i+1)/2-1)],ls[i]
    ls[0],ls[j-1]=ls[j-1],ls[0]
print(ls)
