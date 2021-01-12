from numpy import *
from matplotlib.pyplot import *
from scipy.stats import *

x = array([0, 1, 2, 3, 4, 5])
y = array([0, 0.8, 0.9, 0.1, -0.8, -1])

print(x)
print(y)

p1 = polyfit(x, y, 1)
p2 = polyfit(x, y, 2)
p3 = polyfit(x, y, 3)

print(p1)
print(p2)
print(p3)

xp = linspace(0, 6, 100)
plot(x, y, 'o')
plot(xp, polyval(p1, xp), 'r-')
plot(xp, polyval(p2, xp), 'b-')
plot(xp, polyval(p3, xp), 'y-')
# show()

yfit = (p1[0] * x + p1[1])
print(yfit)
print(y)
yresid = y - yfit
SSR = sum(power(yresid, 2))
TSS = len(y) * var(y)
ESS = TSS-SSR
R2 =ESS/TSS
print(R2)

slope,intercept,r_value,p_value,std_err=linregress(x,y)
print(power(r_value,2))
