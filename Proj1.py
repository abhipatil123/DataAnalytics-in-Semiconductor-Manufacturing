'''
Created on Mar 16, 2018

@author: sps
'''
from __future__ import division
import numpy as np
from scipy import stats
from scipy import spatial
import gaussian
import matplotlib.pyplot as plt
import pandas as pd
import os
headers = ["X","Y"]
#Import Training data
train = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\train_data.csv", engine ="python",header=None)
print train

#Get x and y co-ordinates
x = np.array([float(i) for i in train[0][1:]])
y = np.array([float(i) for i in train[1][1:]]) 
z = np.array([float(i) for i in train[2][1:]]) 

'''
#Import Test data
test = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_data.csv", engine ="python",header=None)
print test
'''
#Generate random locations as training and test data
indices = np.random.permutation(x.shape[0]) #Generates random instances of total_devices
training_idx, test_idx = indices[:300], indices[300:] # Select 250 each out of those
Xtrain, Xtest = x[training_idx], x[test_idx]    
Ytrain, Ytest = y[training_idx], y[test_idx]
#Path for storing results
csv_path = r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\geo_results.csv"
tx = Xtest.reshape(-1,1)
ty = Ytest.reshape(-1,1)
csv_data = np.concatenate((tx,ty),axis = 1)

Ztrain, Ztest = z[training_idx], z[test_idx]
train = np.array(map(lambda x,y,z : [x,y,z] ,Xtrain,Ytrain,Ztrain))
test = np.array(map(lambda x,y : [x,y] ,Xtest,Ytest))

l_to_test = np.arange(1, 200, 2)[1:]
#sigma_to_test = np.arange(0.16, 0.2, 0.04)[1:]
sigma_to_test = [0.12, 0.16,.20, 0.24]
l_opt, sigma_opt, func, rmse_low = gaussian.cross_validate(train,l_values=l_to_test,sigma_values=sigma_to_test,rmse_opt=100000,k_folds=5)
# Create model object from training data
kriging = gaussian.SimpleKriging(training_data=train)

#Return the prediction means for test data from trained model
predict = kriging.predict(test_data=test, l=l_opt, sigma=sigma_opt)
print l_opt,sigma_opt

predict = predict.reshape(-1,1)
Ztest = Ztest.reshape(-1,1)

f_n = 0
csv_data = np.concatenate((csv_data,predict,Ztest),axis=1)
headers = ['X value','Y Value', 'Predicted', 'Actual']
#csv_data = np.insert(csv_data,0,headers)
while(os.path.exists(csv_path)):
    c = csv_path.split("\\")
    c[-1] = "geo_result_%s.csv"%f_n
    csv_path ="\\".join(c)
    f_n += 1
np.savetxt(csv_path, csv_data, delimiter=",",fmt='%s')

