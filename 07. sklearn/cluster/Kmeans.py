import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
digits_train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
digits_test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)
print(digits_test)
test = digits_test[:]
# num_clusters =list(range(1,9))
# inertias =[]
#
# for i in num_clusters:
#   model = KMeans(n_clusters = i)
#   model.fit(digits)
#   inertias.append(model.inertia_)
# plt.plot(num_clusters, inertias, '-o')
#
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
print(type(test))

model = KMeans(n_clusters = 2)
model.fit(digits_train)
labels = model.predict(digits_test)
print(labels)
print(model.cluster_centers_)
digits_test.plot.scatter(x = 3, y=4, c = labels, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()