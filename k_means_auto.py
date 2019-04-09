import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

'''
X = np.array([[1,2],[2,3],[1.5,3],[5,6],[6,7],[5.5,8]])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()


clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r.','g.','c.','b.','k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 25)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidth=5)
plt.show()
'''


df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
#print(df.head())
df.fillna(0, inplace=True)

def handle_non_numeric_data(df):
    columns = df.columns.values
    #print(columns)

    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            column_contents = df[col].values.tolist()
            uniq_elts = set(column_contents)
            x = 0

            for uniq in uniq_elts:
                if uniq not in text_digit_vals:
                    text_digit_vals[uniq] = x
                    x+=1
            #print(convert_to_int(df[col]))           
            df[col] = list(map(convert_to_int, df[col]))
    return df


df = handle_non_numeric_data(df)
#print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])


clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print (correct/len(X))





















    


                
