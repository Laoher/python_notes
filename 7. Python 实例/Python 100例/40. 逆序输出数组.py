import random
ls=[]
for i in range(10):
    ls.append(random.randint(1,100))
print(ls)

for i in range(len(ls)):
    print(ls[-1-i])