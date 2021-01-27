f = open('1.2in1.txt')
lines = list(f)
N=int(lines[0])
a=list(map(int,lines[1].split(' ')))
print(N)
print(a)
#循环

if a is None:
    min = 0
else:
    min = a[0]
    print(min)
    for i in range(N):
        if abs(a[i])<abs(min) or abs(a[i])== abs(min) and min<0 and a[i]>0:
            min = a[i]
print(min)



