# Before we analyze anything, we need to import pandas
import pandas as pd
# 导入/读取csv文件

from numpy.core.defchararray import upper

df = pd.read_csv('shoefly_orders.csv')
# 文件前十行
df.head(10)
# 文件行列数
df.shape()
# 给出表格的许多信息
df.info()
# 选取符合条件的数据
df[(df.shoe_type == 'sandals') & (df.shoe_color == 'black')]  # 这个括号不能省略

# 自己创建表格
# 方法一
df1 = pd.DataFrame({
  'Product ID': [1, 2, 3, 4],
  # add Product Name and Color here
  'Product Name': ['t-shirt', 't-shirt', 'skirt', 'skirt'],
  'Color': ['blue', 'green', 'red', 'black']
})
print(df1)
# 方法二
df2 = pd.DataFrame([
    ['John Smith', '123 Main St.', 34],
    ['Jane Doe', '456 Maple Ave.', 28],
    ['Joe Schmo', '789 Broadway', 51]
    ],
    columns=['name', 'address', 'age'])
# 自己写一个csv文件大概的格式（以下写在csv文件中）
# name,cake_flavor,frosting_flavor,topping
# Devil's Food,chocolate,chocolate,chocolate shavings
# Birthday Cake,vanilla,vanilla,rainbow sprinkles
# Carrot cake,carrot,cream cheese,almonds

# 选取表格的一列数据
clinic_north=df['clinic_north']
# 选取表格的几列作为新的dataframe
new_dataframe=df[['name','age']]
# 选取表格的几行作为新的数据
match=df.iloc[2]  # 比如这个就是选取第三行，第一行是0
match2=df.iloc[2:7]  # 2,3,4,5,6行
# 查看某个参数存在于选项中的行
january_february_march=df[df.month.isin(['January','February','March'])]

# 重置index (从之前的大表中截到的index可能不连续)
df.reset_index()  # 添加了从0开始的index
df.reset_index(drop=True)  # 把之前的index去掉

# 添加不存在的列 直接在中括号里写一个新的列名，后面把内容填上
df['Sold in Bulk?']=['Yes','Yes','No','No']
df['Is taxed?']='Yes'  # 如果所有的行添加的内容都一样的话就直接在后面把相同的这个值给出
df['Sales Tax'] = df.Price * 0.075  # 新增的这个相同的值跟另一列有关系  # 后面也可以是df['Price']
df['Name'] = df.Name.apply(upper)  # 更复杂的操作， 把字母都变成大写 # 使用复杂的公式的话就用 .apply() 括号里放要用的公式
# 如果这个apply没有指定的某一行，括号里要加上 axis=1

# 一行的某一个数据 可以用row.column_name的方式得到
total_earned = lambda row:1.5*row.hourly_wage*(row.hours_worked-40)+400 if row.hours_worked >40 else row.hourly_wage*row.hours_worked

# 改column的名字 You can change all of the column names at once by setting the .columns property to a different list.
df.columns=['ID','Title','Category','Year Released','Rating']
# 方法二
df.rename(columns={'name': 'First Name','age': 'Age'}, inplace=True)
df.rename(columns={'name':'movie_title'},inplace=True)