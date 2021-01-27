import random
from collections import Counter
t=0
total=1000
for i in range(total):
    students = []
    for i in range(53):
        a = random.randint(1,365)
        students.append(a)
    at = Counter(students)
    print(at.values())
    print(Counter(at.values()))
    print(Counter(at.values())[3])
    if 4 in Counter(at.values()):
        t=t+1

print(t/total)


