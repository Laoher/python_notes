# 题目：输入某年某月某日，判断这一天是这一年的第几天？
# 程序分析：以3月5日为例，应该先把前两个月的加起来，然后再加上5天即本年的第几天，特殊情况，闰年且输入月份大于2时需考虑多加一天：

year = int(input("please input the year:\n"))
month = int(input("please input the month:\n"))
day= int(input("please input the day:\n"))

months =[0,31,59,90,120,151,181,212,243,273,304,334]
210
sum = months[month-1]

sum = sum +day

if year%4 ==0 and year%100!=0 or year%400 == 0:
    sum =sum+1

print(sum)