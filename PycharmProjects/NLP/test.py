import string

import gensim as gensim
import numpy as np


#

from nltk.corpus import stopwords
z =[np.array([1,2,3,4,5]),np.array([1,2,3,4,5]),np.array([1,2,5])]
a = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,5]])
b = a=np.random.random((8,3))
c =[a,b]
x = [[1,2,3,4,5],[1,2,3,4,5],[1,2,5,4,5]]
print(x*3)
d = np.asarray(x)
print(d)
print(type(b[0]))
print(b[0])
print(np.array(z))
