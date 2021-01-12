# Naive Bayes classifier is a supervised machine learning algorithm

# 在文字处理这个方面 可以看一个词在不同分类中使用的频率 下次如果这个词出现了那么大概率得出这个属于哪一类的结论

from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer().fit(['apple','organge'])