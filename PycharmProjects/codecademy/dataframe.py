import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")



prod_per_year= df.groupby('year').totalprod.mean().reset_index()

X=prod_per_year.year
print(X)
X = X.values.reshape(-1, 1)
print(X)