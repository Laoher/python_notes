# 导入pyplot的方法
from matplotlib import pyplot as plt

# 最简单的plot的情况
days = [0, 1, 2, 3, 4,5,6]  # 作为x轴的列表
money_spent = [10, 12, 12, 10, 14, 22, 24]
plt.plot(days, money_spent)  # 记得是x的在前面 y的在后面
plt.show()

# It will automatically place the two lines on the same axes and give different colors if call plt.plot() twice
time = [0, 1, 2, 3, 4]
revenue = [200, 400, 650, 800, 850]
costs = [150, 500, 550, 550, 560]

plt.plot(time, revenue)
plt.plot(time, costs)  # 两条线显示在一张图上
plt.show()

# 可以添加在plt.plot(days, money_spent)后面的一些属性
# 颜色 color='purple' color='#82edc9' https://matplotlib.org/users/colors.html
# 线型 linestyle='--' 虚线 https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
# 点标记 marker='s' square https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
plt.axis([0,12,2900,3100])  # 调整x y轴显示的范围 这个表示x轴是0-12 y轴是2900-3100
plt.xlabel('Time')  # x轴的标题
plt.ylabel('Dollars spent on coffee')  # y轴的标题
plt.title('My Last Twelve Years of Coffee Drinking')  # 图标的标题

# 子图 subplot
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
plt.subplots_adjust(wspace=0.2, hspace=0.2)  # 还有top bottom left right
plt.show()

# 图例 legend （图例） 有好几条线在一个图表里的时候就可以用legend标识线条
legend_labels = ['Hyrule','Kakariko', 'Gerudo Valley']  # 按照画的线的顺序, 在画完线之后再进行图例标注
plt.legend(legend_labels,loc =8)

# 也可以直接把label放到plt.plot()里面，但最后还是要加上空的plt.legend()
# loc 等于不同数字对应显示的位置 不加loc的话会自动加在最佳位置
# 0	best
# 1	upper right
# 2	upper left
# 3	lower left
# 4	lower right
# 5	right
# 6	center left
# 7	center right
# 8	lower center
# 9	upper center
# 10	center

# 修改其中一个子图
ax = plt.subplot(1,1,1)  # 如果只有一个图，那么可以用 ax = plt.subplot()
# 并修改这个子图的属性
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep","Oct", "Nov", "Dec"]
months = range(12)
# 标记x轴上的点
ax.set_xticks(months)
# 标记这些点的名字
ax.set_xticklabels(month_names)
# y轴也是一个道理
ax.set_yticks([0.10, 0.25, 0.5, 0.75])
ax.set_yticklabels(["10%", "25%", "50%", "75%"])

# 关闭所有现存的plot
plt.close('all')

# 创建一个宽4英寸，长10英寸的figure
plt.figure(figsize=(4, 10))

# 储存图像
plt.savefig('name_of_graph.png')

# 画出条形图 把plot换成bar
drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales =  [91, 76, 56, 66, 52, 27]
plt.bar(range(len(drinks)),sales)

# side-by-side bars
n = 1  # This is our first dataset (out of 2)
t = 2 # Number of datasets
d = 6 # Number of sets of bars
w = 0.8 # Width of each bar
x_values1 = [t*element + w*n for element
             in range(d)]

n = 2  # This is our second dataset (out of 2)
t = 2 # Number of datasets
d = 6 # Number of sets of bars
w = 0.8 # Width of each bar
x_values2= [t*element + w*n for element in range(d)]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]
plt.bar(x_values1, sales1)
plt.bar(x_values2, sales2)

# 把两种数据叠在一个条上  在plot的函数中加上一个bottom的关键词
plt.bar(range(len(drinks)),sales1)
plt.bar(range(len(drinks)),sales2,bottom = sales1)

# error bar 用来显示误差 后面加一个yerr的关键词
ounces_of_milk = [6, 9, 4, 0, 9, 0]
error = [0.6, 0.9, 0.4, 0, 0.9, 0]
plt.bar(range(len(drinks)), ounces_of_milk, yerr=error,capsize=5)  # capsize 是用来显示横向的条的宽度

# 填充error的范围
revenue = [16000, 14000, 17500, 19500, 21500, 21500, 22000, 23000, 20000, 19500, 18000, 16500]
y_lower=[0.9*i for i in revenue]
y_upper=[1.1*i for i in revenue]
plt.fill_between(months, y_lower, y_upper, alpha=0.2)  # alpha是透明度

# 画饼状图
budget_data = [500, 1000, 750, 300, 100]
budget_categories = ['marketing', 'payroll', 'engineering', 'design', 'misc']
plt.pie(budget_data)
plt.axis('equal')
plt.legend(budget_categories)  # 作为图例的形式
plt.pie(budget_data, labels=budget_categories, autopct='%0.1f%%')  # 在每个扇形旁边标记 扇形上标出数字：一位小数，后面加百分号

# 画出柱状图 histogra 把plot换成hist
# plt.hist(dataset, range=(66,69), bins=40)
# range 是只画66-69的部分，bins代表划分的条数
# 只画出柱状图的轮廓 加属性 histtype='step' # 调整线条的宽度 linewidth = 数字
# 调柱形图的透明度 加属性 alpha=0.5
# 使柱形图的总面积为1 加属性
# 调整线条的宽度 linewidth