from sklearn import tree
# 130 is 130g, 1 is bumpy 0 is not bumpy
features = [[140, 1], [130, 1], [150, 1], [170, 1]]
# 0 is apple and 1 is orange
lables = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,lables)
print(clf.predict([[160, 0]]))
# result is 1 means it think that it is a orange

# even if change all the 1 to 0(bumpy), still result is one