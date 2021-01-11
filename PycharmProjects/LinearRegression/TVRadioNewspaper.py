import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import datasets, linear_model

# read csv file directly from a URL and save the results
data = pd.read_csv('Advertising.csv', index_col=0)

# display the first 5 rows
print(data.head())
print(data.tail())
print(data.shape)

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8)

sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8, kind='reg')

plt.show()
# create a python list of feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['TV', 'Radio', 'Newspaper']]

# print the first 5 rows
print(X.head())

# select a Series from the DataFrame
y = data['Sales']

# equivalent command that works if there are no spaces in the column name
y = data.Sales

# print the first 5 values
print(y.head())

print(type(y))
print(y.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# default split is 75% for training and 25% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)

predictions = {'intercept': linreg.intercept_, 'coefficient': linreg.coef_}

print("Intercept value ", predictions['intercept'])
print("coefficient", predictions['coefficient'])