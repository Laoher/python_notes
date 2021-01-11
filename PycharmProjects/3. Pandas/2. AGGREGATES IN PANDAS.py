import numpy as np
# df.column_name.command() 可以显示所有的command，包括：


# Command	Description
# mean	Average of all values in column
# std	Standard deviation
# median	Median
# max	Maximum value in column
# min	Minimum value in column
# count	Number of values in column
# nunique	Number of unique values in column
# unique	List of unique values in column
# isnull


# .median()可以计算平均数
# print(customers.age)
# >> [23, 25, 31, 35, 35, 46, 62]
# print(customers.age.median())
# >> 35

# .nunique()可以生成所有distinguish的值的数量
# print(shipments.state)
# >> ['CA', 'CA', 'CA', 'CA', 'NY', 'NY', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ', 'NJ']
# print(shipments.state.nunique())
# >> 3

# 计算某一列数据的相关值(成为series)
# df.groupby('column1').column2.measurement()
# 计算某一列数据的相关值(得到DataFrame)
# df.groupby('column1').column2.measurement().reset_index()
# teas_counts = teas.groupby('category').id.count().reset_index()
# 继续修改名字
# teas_counts = teas_counts.rename(columns={"id": "counts"})

# 一些更复杂的计算
# 25%的价格点
# high_earners = df.groupby('category').wage.apply(lambda x: np.percentile(x, 75)).reset_index()

# group by 不止一个值
# shoe_counts=orders.groupby(['shoe_type', 'shoe_color'])['id'].count().reset_index()

# pivot
# shoe_counts = orders.groupby(['shoe_type', 'shoe_color']).id.count().reset_index()
# shoe_counts_pivot = shoe_counts.pivot(columns='shoe_color',index='shoe_type',values='id').reset_index()