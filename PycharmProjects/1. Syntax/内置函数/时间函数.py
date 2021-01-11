import time
from dateutil import parser
print(time.ctime(time.time()))
print(time.asctime(time.localtime(time.time())))
print(time.asctime(time.gmtime(time.time())))

start = time.time()
start2 = time.clock()
for i in range(300000):
    print(i)
end = time.time()  # time是实际的时间会比较大
end2 = time.clock()  # clock是CPU的时间会比较小
print(end - start)  # 1.5418610572814941
print('different is %6.3f' % (end2 - start2))  # 1.469

# 按格式输出时间
dt = parser.parse("Aug 28 2015 12:00AM")
print(dt)  # 2015-08-28 00:00:00
