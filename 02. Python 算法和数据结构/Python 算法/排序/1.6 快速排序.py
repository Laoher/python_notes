# 快速排序quick sort的基本思想：通过一趟排序将待排记录分隔成独立的两部分，
# 其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。

# 快速排序使用分治法来把一个串（list）分为两个子串（sub-lists）。具体算法描述如下：
#
# 从数列中挑出一个元素，称为 “基准”（pivot）；
# 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。
# 在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
# 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

import random

l = []
for i in range(10):
    l.append(random.randint(1, 100))

print(l)
def quicksort(ls):
    if len(ls) == 2:
        if ls[0] > ls[1]:
            ls[0], ls[1] = ls[1], ls[0]
        return ls
    if len(ls) == 1:
        return ls
    if len(ls) == 0:
        return []
    if len(ls)>2:
        t=int(len(ls) / 2)
        pivot = ls[t]
        i=0
        for times in range(len(ls)):
            if ls[i]>pivot and i<t:  # 数字大于参考数且在参考数左边
                    ls.append(ls.pop(i))  # 移到最右
                    t=t-1
            else:
                if ls[i]<pivot and i>t:  # 数字小于参考数且在参考数右边
                    ls.insert(0, ls.pop(i))  # 移到最左
                    t=t+1
                i = i + 1

        return quicksort(ls[:t])+[ls[t]]+quicksort(ls[t+1:])

c=quicksort(l)
print(c)
