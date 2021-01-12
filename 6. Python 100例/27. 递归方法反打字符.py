# 题目：利用递归函数调用方式，将所输入的5个字符，以相反顺序打印出来。

str =list(input("Please input the string:"))
def pr(str):
    if len(str)==1:
        print(str[0])
    else:
        print(str[-1])
        return pr(str[:-1])

pr(str)