import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


clf = neighbors.KNeighborsClassifier()

acc = clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

'''
for i in X_train:
    print (len(i))
    break
'''

ex_val = np.array([[5,1,2,2,1,4,1,8,2]])
ex_val = ex_val.reshape(len(ex_val), -1)


y_pre = clf.predict(ex_val)

##plt.scatter(1, y_pre)
##plt.show()

print(y_pre)
     


     
