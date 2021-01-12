import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_sample=[]
train_label=[]
for i in range(1000):
    single_sample = random.randint(13, 65)
    single_label = 0
    train_sample.append(single_sample)
    train_label.append(single_label)

    single_sample = random.randint(65, 100)
    single_label = 1
    train_sample.append(single_sample)
    train_label.append(single_label)

for i in range(50):
    single_sample = random.randint(13, 65)
    single_label = 1
    train_sample.append(single_sample)
    train_label.append(single_label)

    single_sample = random.randint(65, 100)
    single_label = 0
    train_sample.append(single_sample)
    train_label.append(single_label)

for i in range(len(train_sample)):
    print(train_sample[i])

train_sample = np.array(train_sample)
train_label = np.array(train_label)

scalar  = MinMaxScaler(feature_range =(0,1))
scaled_train_samples = scalar.fit_transform((train_sample).reshape(-1,1))

