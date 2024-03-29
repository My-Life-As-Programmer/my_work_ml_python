import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],[2,3],[1.5,3],[5,6],[6,7],[5.5,8],[2,3.5],
                   [7,8],
                   [3,4]])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()


colors = 10*['g','r','c','b','k']

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False
            if optimized:
                break
            
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for fetureset in clf.classifications[classification]:
        plt.scatter(fetureset[0], fetureset[1], marker="x", color=color, s=150, linewidths=5)


unknows = np.array([[2,3.5],
                   [7,8],
                   [3,4],])
for unknown in unknows:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=color, s=100, linewidths=5)


plt.show()
    
