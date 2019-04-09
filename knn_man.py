import numpy as np
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
from math import sqrt
import pandas as pd
import random

#style.use('fivethirtyeight')

'''
dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_feat = [5,7]

##[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_feat[0], new_feat[1])
##plt.show()
#ecludian_dist = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

#print (ecludian_dist)
'''

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            ecludian_dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([ecludian_dist, group])

    #print (distances)
    votes = [i[1] for i in sorted(distances)[:k]]
    #print (Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
'''
result = k_nearest_neighbors(dataset, new_feat, k=3)
print(result)
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feat[0], new_feat[1], color=result)
plt.show()
'''

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if vote == group:
            correct +=1
        total +=1

print('Accuracy : ',correct/total)
        
        


