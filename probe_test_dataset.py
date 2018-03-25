# Import the `pandas` library as `pd`
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
from sklearn.decomposition import PCA,RandomizedPCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
# Load in the data with `read_csv()`
probe_tests = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_limits.csv", engine ="python",header=None)
# Print out `digits`
#print(probe_tests)
#print(probe_tests.loc[1:2396,3:]) 
device = [True for i in range(0,2394)]

for j in range(1,2395):
    for i in range(3,4):
        if not ((float(probe_tests.loc[j,i]) > float(test_limits.loc[i-2,1])) 
        and (float(probe_tests.loc[j,i]) < float(test_limits.loc[i-2,2]))):
            print j,float(probe_tests.loc[j,i])
            print float(probe_tests.loc[j,1]),float(probe_tests.loc[j,2])
            device[j-1] = False
            break
c = 0;
for j in range(0,2394): 
    print device[j]      
    if not device[j]:
        c +=1
print "failed %d"%(c)

z = list(probe_tests[3][1:2395]) # Probe-test 1 measurement
x = list(probe_tests[1][1:2395]) # x- co-ordinate
y = list(probe_tests[2][1:2395]) # y- co-ordinate

X_Input_Coordinates = np.array(zip(x,y))


#print(KMeans.labels_[::10])
'''
# Create a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(probe_tests.loc[1:1,3:])

# Inspect the shape
reduced_data_pca.shape

print(reduced_data_pca)

'''

    
'''    
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

trace =  go.Heatmap(z = list(probe_tests[103][1:2396]),
                    x = list(probe_tests[1][1:2396]),
                    y = list(probe_tests[2][1:2396]))

data = [trace]
py.iplot(data, filename = 'Wafer Heatmap')
'''

