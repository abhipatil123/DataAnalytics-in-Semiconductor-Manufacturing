'''
Created on Mar 25, 2018

@author: sps
'''
from __future__ import division
import numpy as np
from scipy import stats
from scipy import spatial
import gaussian_1
import matplotlib.pyplot as plt
import pandas as pd
import os


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.cluster import KMeans



def progressive_sampling(train,test,std,P1test):
    d = spatial.distance_matrix(test, train[:,:-1])
    min_dist =[]
    for i in range(len(d)):
        #j, = np.where(d[i] == min(d[i]))
        min_dist.append(max(d[i]))
    std = std.flatten()
    min_dist = np.array(min_dist)
    sorted_dist_id = np.flip(min_dist.argsort(), axis = 0)
    max_var = -1000
    max_var_id = sorted_dist_id[-1]
    p = int(0.2 *len(sorted_dist_id)) # 20 % of min distances
    for i in sorted_dist_id[:p]:
        if max_var < std[i]:
            max_var = std[i]
            max_var_id = i
    new_training_sample = np.append(test[max_var_id],P1test[max_var_id])
    train = np.append(train,new_training_sample).reshape(-1,len(test[0])+1)
    test = np.delete(test, max_var_id, axis=0)
    P1test = np.delete(P1test, max_var_id, axis=0)
    std = np.delete(std, max_var_id, axis=0)
    return train,test,std,P1test
    
probe_tests = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv(r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_limits.csv", engine ="python",header=None)
csv_path = r"C:\Users\Abhishek\Dropbox\II Sem\Special topics-data analytics\Project-1\test_results.csv"
#total_devices = 54935
total_devices = 2394
total_test = 102        
device = []

'''
for i in range(0,total_test):
    l = [False for j in range(0,total_devices)]
    device.append(l)
devices_passed = [True for i in range(0,total_devices)]
'''
f_n = 0
x = np.array([float(i) for i in probe_tests[1][1:total_devices+1]])
y = np.array([float(i) for i in probe_tests[2][1:total_devices+1]]) 
radial = True
kmean = False
for pb_idx in range(103,104):
    '''
    indices = np.random.permutation(len(low_confidence_ids))
    train_idx = indices[len(indices)]
    test_idx = [i for i in test_idx if not i in indices]
    #training_idx = 
    '''
    probe_test = np.array([float(i) for i in probe_tests[pb_idx][1:total_devices+1]])

    X = probe_test.reshape(-1,1)
    if kmean:
    
        # Number of clusters
        kmeans = KMeans(n_clusters=3)
        # Fitting the input data
        kmeans = kmeans.fit(X)
        # Getting the cluster labels
        labels = kmeans.predict(X)
        # Centroid values
        centroids = kmeans.cluster_centers_
        score_opt = 0
        K_opt = 0
        metrics.calinski_harabaz_score(X, labels)
        for k in range(2,5):
            kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
            labels = kmeans_model.labels_
            score =  metrics.calinski_harabaz_score(X, labels)
            print score
            if score_opt < score:
                score_opt = score
                k_opt = k
                labels_opt= kmeans_model.labels_
                centroids_opt = kmeans_model.cluster_centers_
        
        data_s = {}
        for index,i in zip(range(0,len(X.flatten())),labels_opt):
            if i not in data_s.keys():
                data_s[i]=[]
            data_s[i].append(np.array([x[index],y[index],probe_test[index]]))
    else: # For each probetest map a key_number to the x,y and probe values
        data_s ={"0":np.array(map(lambda x,y,z : [x,y,z] ,x,y,probe_test))}
    
    #data_s = {"0":}
    
    '''
    for i in range(0,total_test):
        l = [False for j in range(0,total_devices)]
        device.append(l)
    devices_passed = [True for i in range(0,total_devices)]
    '''
    sampling_steps = 10
    sample_perc = 0.01
    for i in data_s.keys():
        data = np.array(data_s[i])
        x,y,probe_test1= data[:,:1].flatten(),data[:,1:2].flatten(),data[:,-1:].flatten() #Flatten - make array 1D
        lt = len(data)
        ti = int(sample_perc*lt) #1per of total devices
        indices = np.random.permutation(lt)
        training_idx, test_idx = indices[:ti], indices[ti:]
        Xtrain, Xtest = x[training_idx], x[test_idx]
        Ytrain, Ytest = y[training_idx], y[test_idx]
        P1train, P1test = probe_test1[training_idx], probe_test1[test_idx]
        if radial:
            Rtest = (Xtest**2 + Ytest**2)**0.5
            Rtrain = (Xtrain**2 + Ytrain**2)**0.5
            train = np.array(map(lambda x,y,r,z : [x,y,r,z] ,Xtrain,Ytrain,Rtrain,P1train))
            test = np.array(map(lambda x,y,r : [x,y,r] ,Xtest,Ytest,Rtest))
        else:
            train = np.array(map(lambda x,y,z : [x,y,z] ,Xtrain,Ytrain,P1train))
            test = np.array(map(lambda x,y : [x,y] ,Xtest,Ytest))
        
        
        for step in range(0,sampling_steps):
            l_to_test = np.arange(1, 50, 5)[1:]
            #sigma_to_test = np.arange(0.16, 0.2, 0.04)[1:]
            sigma_to_test = [0,0.0001,.01,.5]
            l_opt, sigma_opt, func, rmse_low = gaussian_1.cross_validate(train,l_values=l_to_test,sigma_values=sigma_to_test,rmse_opt=100000,k_folds=5)
            # Create model object from training data
            kriging = gaussian_1.SimpleKriging(training_data=train,return_std = True)
            
            # Return prediction means for test data from trained model
            
            predict,std = kriging.predict(test_data=test, l=l_opt, sigma=sigma_opt)
            
            if step == sampling_steps - 1:
                break
            for t in range(ti):
                train,test,std,P1test = progressive_sampling(train,test,std,P1test)
                
        headers = ["X","Y"]
        tx = np.append(test[:,0],["","passed" ,"failed" ,"yield_Loss","test_escapes","length_scale_opt","sigma_opt"]).reshape(-1,1)
        ty = np.append(test[:,1],["","" ,"" ,"","","",""]).reshape(-1,1)
        csv_data = np.concatenate((tx,ty),axis = 1)
        
        y = test[:,1]
        x = test[:, 0]
        
        plt.scatter(x, y, c=predict.T.flatten())
        plt.colorbar(ticks=[np.min(predict), np.max(predict)], label='Probe_test')
        #plt.show()
        print l_opt,sigma_opt
        test_limit_min, test_limit_max = float(test_limits.loc[pb_idx-2,1]),float(test_limits.loc[pb_idx-2,2])
        
        actual_passed = 0
        predicted_passed = 0
        escaped_test = 0
        
        P1test = P1test.reshape(-1,1)
        low_confidence_ids = []
        actual_passed = 0
        predicted_passed = 0
        escaped_test = 0
        for id in range(len(P1test)):
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
                    low_confidence_ids.append([id,test[id][0],test[id][1]])
            '''
            if ((predict[id][0] - 1.96 * std[id][0] > test_limit_min) 
            and (predict[id][0] + 1.96 * std[id][0] < test_limit_max)):
                if passed:
                    predicted_passed +=1 
                else:
                    escaped_test += 1
                    low_confidence_ids.append([id,test[id][0],test[id][1]])
            '''
        if not(actual_passed):
            actual_passed = 1
        if not(predicted_passed):
            predicted_passed = 1 # for div by zero          
        actual_failed = len(predict) - actual_passed  #len(predict) total no of test devices
        predicted_failed = len(predict) - predicted_passed - escaped_test                 
        P1test =  np.append(P1test,["",actual_passed ,actual_failed ,float(actual_failed)*100/actual_passed ,"","",""]).reshape(-1,1) 
        predict =  np.append(predict,["",predicted_passed ,predicted_failed ,float(predicted_failed)*100/predicted_passed,escaped_test,l_opt,sigma_opt]).reshape(-1,1)
        csv_data = np.concatenate((csv_data, P1test, predict),axis=1)
        headers += [str(probe_tests[pb_idx][0])+'_actual_cluster_'+str(i),str(probe_tests[pb_idx][0])+'_predicted_cluster_'+str(i)]
        print "test_no:",pb_idx
        print escaped_test
    
        csv_data = np.insert(csv_data,0,headers).reshape(-1,len(headers)).astype("string")
        while(os.path.exists(csv_path)):
            c = csv_path.split("\\")
            c[-1] = "test_result_%s.csv"%f_n
            csv_path ="\\".join(c)
            f_n += 1
        np.savetxt(csv_path, csv_data, delimiter=",",fmt='%s')

    