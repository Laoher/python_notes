# 题目：利用条件运算符的嵌套来完成此题：学习成绩>=90分的同学用A表示，60-89分之间的用B表示，60分以下的用C表示。
#
# 程序分析：程序分析：(a>b)?a:b这是条件运算符的基本例子。

score =int(input("please input the score:"))

if score>90:
    grade ="A"
elif score>80:
    grade ="B"
else:
    grade ="F"

print(grade)