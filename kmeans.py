from __future__ import division
import matplotlib.pyplot as plt
import pylab as p
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.preprocessing.data import StandardScaler
import numpy as np
from scipy import stats
from scipy import spatial
import matplotlib.pyplot as plt
import pandas as pd

total_devices = 2395
total_test = 102

# Load in the data with `read_csv()`
probe_tests = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\probe_tests.csv", engine ="python",header=None)

x = list((probe_tests[1][1:])) # x- co-ordinate
x = map(float, x)
y = list((probe_tests[2][1:])) # y- co-ordinate
y = map(float, y)
z = list((probe_tests[3][3:])) # Probe-test 1 measurement
z = map(float , z)

cluster_values_each_measurement = []
k_values = []
for pb_idx in range(3,4):
    probe_test = np.array([float(i) for i in probe_tests[pb_idx][1:total_devices+1]])
    Input_coordinates = np.array(map(list, zip(x,y,probe_test)))   
    #Input_coordinates = preprocessing.normalize(Input_coordinates, norm='l2')
    # Creating a sample dataset with 4 clusters
    #X, y = make_blobs(n_samples=800, n_features=3, centers=4)
    #plt.scatter(Input_coordinates[:,0], Input_coordinates[:,1], Input_coordinates[:,2],label = 'True Position')
    #plt.show()
    
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Input_coordinates[:, 0],Input_coordinates[:, 1], Input_coordinates[:, 2])
    p.show()
    '''
    
    # Calculating the CH score for a given label(k-value)
    # Higher the score means the clusters are dense and well separated
    k =[]
    ch =[]
    for i in range(2,6):
        kmeans = KMeans(n_clusters=i)  
        kmeans.fit(Input_coordinates)
        
        C = kmeans.cluster_centers_ 
        labels = kmeans.labels_
        
        CH_Score = metrics.calinski_harabaz_score(Input_coordinates, labels) 
        k.append(i)
        ch.append(CH_Score)
    
    
    #plt.scatter(k,ch,label = 'CH score for k values')
    #plt.show()
    
    maxi = 0
    for i in range(0,4):
        if ch[i] > maxi:
            maxi = ch[i]
            k_opt_value = i+2
    print k_opt_value
    
    kmeans = KMeans(n_clusters=k_opt_value)  
    kmeans.fit(Input_coordinates)
    k_values.append(k_opt_value)
    
    {i: Input_coordinates[np.where(kmeans.labels_ == i)] for i in range(kmeans.n_clusters)}
    #Print and plot each cluster points separately
    
    cluster_values = []
    for i in set(kmeans.labels_): 
        index = kmeans.labels_ == i
        #print index # indicate points where label is i
        #print Input_coordinates[index] #List of lists of cluster points of cluster i
        cluster_values.append(Input_coordinates[index])
        #plt.plot(Input_coordinates[index,0], Input_coordinates[index,1], 'o')
    #plt.show()
    cluster_values_each_measurement.append(cluster_values)
    #3d Plot after Clustering
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Input_coordinates[:, 0], Input_coordinates[:, 1], Input_coordinates[:, 2], c=kmeans.labels_,
                                                                                             cmap='rainbow')
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
    p.show()
    #plt.scatter(Input_coordinates[:,0],Input_coordinates[:,1], c=kmeans.labels_, cmap='rainbow') 
    #plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='black')   
    #plt.show()
    '''
print cluster_values_each_measurement
print len(cluster_values_each_measurement[0])