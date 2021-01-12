# 不只是鸡和兔, 任何动物有任何的头和脚都可以计算, 暂时不考虑抬杠的情况, 并未进行深层程序调试

class chick:
    head =1
    feet =2

class rabit:
    head =1
    feet=4

def solve(total_head, total_feet):
    x= (rabit.head*total_feet-rabit.feet*total_head)/(rabit.head*chick.feet-rabit.feet*chick.head)
    y= (total_head-chick.head*x)/rabit.head
    return x,y

chicks, rabits = solve(3,12)
print('%i' % chicks,'%i' % rabits)