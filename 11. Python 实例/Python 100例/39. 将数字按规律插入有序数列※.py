# 题目：有一个已经排好序的数组。现输入一个数，要求按原来的规律将它插入数组中。
#
# 程序分析：首先判断此数是否大于最后一个数，然后再考虑插入中间的数的情况，插入后此元素之后的数，依次后移一个位置。
import random

ls=[]
for i in range(10):
    ls.append(random.randint(1,100))
ls.sort()
print(ls)

# b=int(input())
b=18
min = 0
max = int(len(ls))-1
if b >= ls[max]:
    ls.insert(max + 1, b)
elif b <= ls[min]:
    ls.insert(min, b)
else:
    while max-min>0:
        if max-min==1:
            ls.insert(max,b)
            max=min
        else:
            if b>ls[int(min+(max-min)/2)]:
                min=min+int((max-min)/2)
            elif b<ls[int(min+(max-min)/2)]:
                max=min+int((max-min)/2)
            else:
                ls.insert(int(min+(max-min)/2),b)
                break

print(ls)