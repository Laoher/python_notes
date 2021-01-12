import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model
import linecache
import statistics

# read csv file directly from a URL and save the results
data = pd.read_csv('Advertising.csv', index_col=0)

# the_line = linecache.getline('Advertising.csv', 22)
# print(the_line)

# print(data)
# l = [x[0:] for x in data[100:]]
#
# print(l)



# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8)

sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8, kind='reg')

# plt.show()

# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]

# print(np.mean(X))


#
# print(np.std(X))

# select a Series from the DataFrame
y = data['Sales']
# z = data['TV']
# print(np.cov(y,z))
print(statistics.median(y))
linreg = linear_model.LinearRegression()
linreg.fit(X, y)

predictions = {'intercept': linreg.intercept_, 'coefficient': linreg.coef_}

print("Intercept value ", predictions['intercept'])
print("coefficient", predictions['coefficient'])
