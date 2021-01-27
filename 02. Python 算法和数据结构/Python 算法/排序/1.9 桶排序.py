# 桶排序(Bucket sort)是计数排序的升级版。它利用了函数的映射关系，高效与否的关键就在于这个映射函数的确定。
# 桶排序的工作的原理：假设输入数据服从均匀分布，将数据分到有限数量的桶里，
# 每个桶再分别排序（有可能再使用别的排序算法或是以递归方式继续使用桶排序进行排）。

# 算法描述
# 设置一个定量的数组当作空桶；
# 遍历输入数据，并且把数据一个一个放到对应的桶里去；
# 对每个不是空的桶进行排序；
# 从不是空的桶里把排好序的数据拼接起来。
import random

ls = []
for i in range(100):
    ls.append(random.randint(1, 99))

print(ls)
count={}
for i in range(10):
    count[i]= []

for i in ls:
    a=i//10
    count[a].append(i)


print(count)

for i in count:
    count[i].sort()
print(count)

ls=[]
for i in count:
    for j in range(len(count[i])):
        ls.append(count[i][j])

print(ls)