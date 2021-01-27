# 题目：输出 9*9 乘法口诀表。
# 程序分析：分行与列考虑，共9行9列，i控制行，j控制列。

# 1*1 = 1
# 1*2 = 2 2*2 = 4
# 1*3 = 3 2*3 = 6 3*3 = 9
for j in range(1,10):
    for i in range(1,10):
        if i<=j:
            print(str(i)+"*"+str(j)+" = "+str(j*i)+" ",end="")
    print("\n")


