# 题目：一个5位数，判断它是不是回文数。即12321是回文数，个位与万位相同，十位与千位相同。

a =11134543111

ls =list(str(a))
re =1
for i in range(len(ls)//2):
    if ls[i]!=ls[-i-1]:
        re =0
        print("not 回文")
        break
if re ==1:
    print("回文")

