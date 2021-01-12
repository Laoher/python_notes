import numpy as np
from collections import Counter

# assume all 3 girls are born on 8/8 and there are 365 days in a year
def find_position(a_list):
    for i in range(len(a_list)):
        if a_list[len(a_list)-i-1] != 1:
            digit = len(a_list)-i-1
            break
    return digit
def C(m,n):
    return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))

# get different combination

list_all =[]

max = 50
list_temp = [max] # 7
list_all.append(list_temp)
# list_all.append(list_temp)
while list_temp[0] != 1:
    position =find_position(list_temp) # 1
    # print(position)


    list_initial = list_temp[:position]  # []
    restart = list_temp[position]-1  # 4
    list_initial.append(restart) # [4,1]
    if restart !=1:
        for i in range(restart):
            i = restart-i
            if sum(list_initial)+i> max:
                continue
            elif sum(list_initial)+i== max:
                list_initial.append(i)
                list_temp=list_initial
                list_all.append(list_temp)
                break
            else:
                while(sum(list_initial)+i<=max):
                    list_initial.append(i)
                if sum(list_initial) == max:
                    list_temp = list_initial
                    list_all.append(list_temp)

    else:
        diff = max - sum(list_initial)
        for i in range(diff):
            list_initial.append(1)
        list_temp = list_initial
        list_all.append(list_temp)

list_real = []
for i in list_all:
    if 3 not in i:
        list_real.append(i)

print(list_real)
print(len(list_real))

sum = 0
num = 0
for i in list_real:
    prob = 0
    list_sum = 1
    num = num + 1
    for j in i:
        prob = ((365-num)/365)*((1/365)**(j-1))
        list_sum = list_sum*prob
    sum = sum + list_sum

print(sum/365/365/C())