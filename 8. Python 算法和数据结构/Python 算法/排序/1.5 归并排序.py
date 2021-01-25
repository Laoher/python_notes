# 归并排序(Merge Sort)是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。
# 将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。
# 若将两个有序表合并成一个有序表，称为2 - 路归并。
#
# 算法描述
# 把长度为n的输入序列分成两个长度为n / 2
# 的子序列；
# 对这两个子序列分别采用归并排序；
# 将两个排序好的子序列合并成一个最终的排序序列。
import random

s = []
for i in range(10):
    s.append(random.randint(1, 100))

print(s)


def recursort(ls):
    if len(ls) > 2:
        return merge(recursort(ls[:int(len(ls) / 2)]), recursort(ls[int(len(ls) / 2):]))
    if len(ls) == 2:
        if ls[0] > ls[1]:
            ls[0], ls[1] = ls[1], ls[0]
        return ls
    if len(ls) == 1:
        return ls


def merge(a, b):
    ls = []
    i = j = 0
    for number in range(len(a) + len(b)):
        if i >= len(a):
            ls.append(b[j])
            j = j + 1
        elif j >= len(b):
            ls.append(a[i])
            i = i + 1
        else:
            if a[i] < b[j]:
                ls.append(a[i])
                i = i + 1
            else:
                ls.append(b[j])
                j = j + 1
    return ls
l = recursort(s)
print(l)
