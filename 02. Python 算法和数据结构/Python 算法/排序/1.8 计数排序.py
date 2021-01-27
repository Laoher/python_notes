# 计数排序(Counting Sort)不是基于比较的排序算法，其核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。 作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。
# 就真的很类似于桶排序
#算法描述
# 找出待排序的数组中最大和最小的元素；
# 统计数组中每个值为i的元素出现的次数，存入数组C的第i项；
# 对所有的计数累加（从C中的第一个元素开始，每一项和前一项相加）；
# 反向填充目标数组：将每个元素i放在新数组的第C(i)项，每放一个元素就将C(i)减去1。

import random
ls = []
for i in range(20):
    ls.append(random.randint(1, 10))

print(ls)

print(max(ls),min(ls))

count= {key:value for key, value in zip(list(range(min(ls),max(ls)+1)), [0]*(max(ls)-min(ls)+1))}
print(count)

for i in ls:
    count[i]=count[i]+1

print(count)
ls=[]
for i in count:
    for j in range(count[i]):
        ls.append(i)

print(ls)