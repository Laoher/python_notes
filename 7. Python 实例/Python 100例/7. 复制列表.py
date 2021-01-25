# 题目：将一个列表的数据复制到另一个列表中。

# 程序分析：使用列表[:]。

ls1 = [1,2,3,4]
ls2 = [1,2,2]
for i in range(len(ls1)):
    ls2.append(ls1[i])
print(ls2)

