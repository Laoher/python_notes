def varfunc():
    var = 0
    print('var = %d' % var)
    var += 1


for i in range(3):
    varfunc()  # 每次调用都被重新赋值 所以var一直是 0

# 类的属性
# 作为类的一个属性吧
class Static:
    StaticVar = 5
    def __init__(self, fruit):
        self.fruit = fruit
        print(self.fruit)
    def varfunc(self):
        self.StaticVar += 1
        print(self.StaticVar)

print(Static.StaticVar)
a = Static("banana")
print(a.fruit)
for i in range(3):
    a.varfunc()

t=Static("apple")  # 新建的对象初始的值还是 5
t.varfunc()  # 经过一轮function之后变成了 6
print(a.StaticVar)  # 之前的对象还是 8

t.__init__("peach")
print(t.fruit)