# 基数排序Radix Sort
# 是按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。
# 有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。
# 最后的次序就是高优先级高的在前，高优先级相同的低优先级高的在前。

# 算法描述
# 取得数组中的最大数，并取得位数；
# arr为原始数组，从最低位开始取每个位组成radix数组；
# 对radix进行计数排序（利用计数排序适用于小范围数的特点）；

import random
ls = []
for i in range(10):
    ls.append(random.randint(1, 99))

print(ls)


last={}
for i in range(10):
    last[i]= []

for i in ls:
    a=i%10
    last[a].append(i)
print(last)

first={}
for i in range(10):
    first[i]= []
print(last)
for i in range(10):
    for j in range(len(last[i])):
        a=last[i][j]//10
        first[a].append(last[i][j])
print(first)

ls=[]
for i in range(10):
    for j in range(len(first[i])):
        a=first[i][j]//10
        ls.append(first[i][j])
print(ls)