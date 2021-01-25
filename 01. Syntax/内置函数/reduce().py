from functools import reduce
# reduce起到一个累计运算的作用 以后可能会有更复杂的运算
reduce(lambda x, y: x+y, [1,2,3,4,5])  # 使用 lambda 匿名函数  # 15
