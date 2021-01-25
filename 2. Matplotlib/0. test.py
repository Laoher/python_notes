from matplotlib import pyplot as plt

# 最简单的plot的情况
days = [0, 1, 2, 3, 4,5,6]  # 作为x轴的列表
money_spent = [10, 12, 12, 10, 14, 22, 24]
plt.subplot(3, 1, 1)  #先画出
plt.plot(days, money_spent)
plt.subplot(3,2,3)
plt.plot(days, money_spent)
plt.subplot(3,3,6)
plt.plot(days, money_spent)
plt.subplot(3,2,5)
plt.plot(days, money_spent)
plt.subplot(3,3,9)
plt.plot(days, money_spent)
plt.subplots_adjust(wspace=0.01)
plt.show()
