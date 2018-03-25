# Import the `pandas` library as `pd`
import pandas as pd
from sklearn.decomposition import PCA,RandomizedPCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Load in the data with `read_csv()`
probe_tests = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_limits.csv", engine ="python",header=None)


total_devices = 54935
total_tests = 102
device =[]
for i in range(0,total_devices):  #device[][] = 54935 rows(no of devices) and 103 columns (probe_tests for each device)
    l = [False for j in range(0,total_tests)]
    device.append(l)

for j in range(1,total_devices-1):
    for i in range(3,total_tests+3):
        if ((float(probe_tests.loc[j,i]) > float(test_limits.loc[i-2,1])) 
        and (float(probe_tests.loc[j,i]) < float(test_limits.loc[i-2,2]))):
            device[j-1][i-3] = True

        
for i in range(0,total_tests-1):
    c =0
    for j in range(0,total_devices-1):       
        if device[j][i]:
            c +=1
    print "%f percentage devices passing test %d"%(c*100/float(total_devices),i)

d=0
for i in range(0,total_devices-1):
    c =0
    for j in range(0,total_tests-1):
        if not device[i][j]:
            c+=1;
    if (c>0): #If device fails anyone of test count that device as fail
        d+=1;
print "percentage of failed devices is %f"%(d*100/float(54935))




import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

trace =  go.Heatmap(z = list(probe_tests[3][1:2396]),
                    x = list(probe_tests[1][1:2396]),
                    y = list(probe_tests[2][1:2396]))

data = [trace]
py.iplot(data, filename = 'Wafer Heatmap')

