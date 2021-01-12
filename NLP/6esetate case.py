import pandas as pd
import jieba
dataset = pd.read_csv('work.csv', sep=',', names=['label','sentence']).astype(str)
print(dataset)
print(type(dataset['label']))
cw = lambda x: list(jieba.cut(x))
dataset['words'] = dataset['sentence'].apply(cw)
print(dataset)