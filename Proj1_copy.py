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
import math
from kmeans import Input_coordinates,cluster_values,cluster_values_each_measurement,k_values

probe_tests = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_limits.csv", engine ="python",header=None)
csv_path = r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_results.csv"
folder_path = r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1"
#total_devices = 54935
total_devices = 2395
total_test = 102
device = []

'''
for i in range(0,total_test):
    l = [False for j in range(0,total_devices)]
    device.append(l)
devices_passed = [True for i in range(0,total_devices)]
'''
f_n = 0
#x = np.array([float(i) for i in probe_tests[1][1:total_devices+1]])
#y = np.array([float(i) for i in probe_tests[2][1:total_devices+1]]) 


#Get radial values as input
#r = np.array([math.sqrt(x[i]*x[i] + y[i]*y[i]) for i in range(0,total_devices)])
#print x
#print y

#RTrain, RTest = r[training_idx], r[test_idx]


csv_data = []
#For all Probetest measurements
for pb_idx in range(103,104):   
    #Create a new folder
    newpath = r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1" 
    c = newpath.split("\\")
    c.append("test_result_%s"%pb_idx)
    newpath ="\\".join(c)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    f_n=0
    #Loop for each cluster for each measurement
    for k in range(0,k_values[pb_idx-103]):
        x = np.array(cluster_values_each_measurement[pb_idx-103][k][:,0]) # Get x coordinates from second cluster
        y = np.array(cluster_values_each_measurement[pb_idx-103][k][:,1]) # Get y coordinates from second cluster
        
        #coords = np.concatenate((x, y), axis=0)
        
        bounding_box = [x.min(), x.max(), y.min(), y.max()]
        #Generate random locations as training and test data
        indices = np.random.permutation(x.shape[0]) #Generates random instances of total_devices
        train_len = int(math.floor((10/100)*len(x))) # Select 25% as training data
        training_idx, test_idx = indices[:train_len], indices[train_len:] 
        Xtrain, Xtest = x[training_idx], x[test_idx]    
        Ytrain, Ytest = y[training_idx], y[test_idx]
        headers = ["X","Y"]
        tx = np.append(Xtest,["","passed" ,"failed" ,"yield_Loss","test_escapes","length_scale_opt","sigma_opt"]).reshape(-1,1)
        ty = np.append(Ytest,["","" ,"" ,"","","",""]).reshape(-1,1)     
        
        #tr = np.append(RTest,["","","","","","",""]).reshape(-1,1)
        csv_data = np.column_stack((tx,ty))
        #probe_test1 = np.array([float(i) for i in probe_tests[pb_idx][1:total_devices+1]])
        probe_test1 = np.array([float(i) for i in np.array(cluster_values_each_measurement[pb_idx-103][k][:,2])])
        #probe_test1 = np.array(cluster_values[0][:,2])
        #Map all the x,y and probetest values interms of training and test data
        P1train, P1test = probe_test1[training_idx], probe_test1[test_idx]
        train = np.array(map(lambda x,y,z : [x,y,z] ,Xtrain,Ytrain,P1train))
        test = np.array(map(lambda x,y : [x,y] ,Xtest,Ytest))
    
        l_to_test = np.arange(1, 200, 2)[1:]
        #sigma_to_test = np.arange(0.16, 0.2, 0.04)[1:]
        sigma_to_test = [0.12, 0.16,.20, 0.24]
        #l_opt, sigma_opt, func, rmse_low = gaussian.cross_validate(train,l_values=l_to_test,sigma_values=sigma_to_test,rmse_opt=100000,k_folds=5)
        # Create model object from training data
        l_opt = 199
        sigma_opt = 0.24
        kriging = gaussian.SimpleKriging(training_data=train)
        
        # Return prediction means for test data from trained model
        
        predict = kriging.predict(test_data=test, l=l_opt, sigma=sigma_opt)
        print l_opt,sigma_opt
        test_limit_min, test_limit_max = float(test_limits.loc[pb_idx-2,1]),float(test_limits.loc[pb_idx-2,2])
        
        actual_passed = 0
        predicted_passed = 0
        escaped_test = 0
        
        P1test = P1test.reshape(-1,1)
        
        for id in range(len(test_idx)):
            passed = False
            
            if (((P1test[id][0]) > test_limit_min) 
            and ((P1test[id][0]) < test_limit_max)):
                actual_passed +=1 
                passed = True
                       
            if ((predict[id][0] > test_limit_min) 
            and (predict[id][0] < test_limit_max)):
                if passed:  
                    predicted_passed +=1 
                else:
                    escaped_test += 1
                    
        actual_failed = len(predict) - actual_passed  #len(predict) total no of test devices
        predicted_failed = len(predict) - predicted_passed - escaped_test
        if (actual_passed>0) and (predicted_passed>0):                 
            P1test =  np.append(P1test,["",actual_passed ,actual_failed ,float(actual_failed)*100/actual_passed ,"","",""]).reshape(-1,1) 
            predict =  np.append(predict,["",predicted_passed ,predicted_failed ,float(predicted_failed)*100/predicted_passed,escaped_test,l_opt,sigma_opt]).reshape(-1,1)
            csv_data = np.concatenate((csv_data, P1test, predict),axis=1)
            headers += [str(probe_tests[pb_idx][0])+'_actual',str(probe_tests[pb_idx][0])+'_predicted']
            print "test_no:",pb_idx
        
        # Store a different csv file for each cluster values
        csv_path = newpath
        csv_data = np.insert(csv_data,0,headers).reshape(-1,len(headers)).astype("string")
        #while(os.path.exists(csv_path)):
        c = csv_path.split("\\")
        c.append("test_result_%s_%s.csv"%(pb_idx,f_n))
        csv_path ="\\".join(c)
        f_n += 1
        np.savetxt(csv_path, csv_data, delimiter=",",fmt='%s')
        print f_n
        
#viz = kriging.simulate(bbox=bounding_box, ncells=50, l=5, sigma=0.2, indices=True, show_visual=True)