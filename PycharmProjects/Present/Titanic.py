import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

data=pd.read_csv('/Users/tyler/PycharmProjects/Present/train.csv')

data['initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')


print(pd.crosstab(data.Sex,data.Initial))