import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
style.use('ggplot')


X = np.array([
             [1,2],
             [1.5,1.8],
             [5,8],
             [8,8],
             [1,0.6],
             [9,11]
             ]
            )

plt.scatter(X[:,0] , X[:,1], s=150)
plt.show()

colors = ["g","r","c","b","k","y"]

# tolerance - how much cerntroid is going to move
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}
# initialize the first 2 data points as centroids for 2 clusters as a start
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classifications= {}  # centroids and clasifications
# values will be features set, key will be centroids
            for i in range(self.k):
                self.classifications[i] = []    # start with empty list
                
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid])
                                            for centroid in self.centroids]
                # above will have the distanes for each centroid in 2 lists here
                classification = distances.index(min(distances))
# we are clearing out classifications for every iteration and append the feature 
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
##                pass
                self.centroids[classification] = np.average(self.classifications[classification],axis=0) 

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
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
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='o',color='k',s=150,linewidth=5)

for classification in clf.classifications:
    color = colors[classification]
##    print('color: ',color, ' classifi: ', classification, ' color : ', colors )
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker='x',color=color,s=150,linewidth=5)

unknowns = np.array([[3,4],
                     [9,2],
                     [1,6],
                     [4,2.4],
                     [4.6,7.8]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidths=5)

plt.show()




















    
