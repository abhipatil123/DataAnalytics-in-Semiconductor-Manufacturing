'''
Created on Mar 8, 2018

@author: amp
'''

from __future__ import division
import numpy as np
from scipy import stats
from scipy import spatial
import pandas as pd
import os
import numpy as np
import GP
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import std, indices
from matplotlib.pyplot import ylabel, xlabel
from sklearn import metrics
from sklearn.cluster import KMeans
from prompt_toolkit.utils import is_windows



def get_mean(data =[]):
    try:
        return sum(data)/float(len(data))
    except Exception as e:
        return None
    
def get_kmean_clusters_data(X,xr=np.array([]),yr=np.array([]),pb=np.array([]),clusters = None,return_orig_indices = False):
    """
    k mean clustering on given data set
    
    """
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
    if clusters:
        min = clusters
        max = clusters + 1
    else:
        min = 2
        max = 5
    for k in range(min,max):
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
    data_s_orig_indices = {}
    for index,i in zip(range(0,len(X.flatten())),labels_opt):
        if i not in data_s.keys():
            data_s[i]=[]
            data_s_orig_indices[i] = []
            
        if xr.any() and yr.any() and pb.any():
            data_s[i].append(np.array([xr[index],yr[index],pb[index]]))
        else:
            data_s[i].append(X[index])
        data_s_orig_indices[i].append(index)
            
    if not return_orig_indices:
        return data_s,centroids_opt
    
    return data_s ,centroids_opt,data_s_orig_indices 

def create_path_if_not_exists(result_folder):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
        
def train_test_split(data, ti, kmean = True, radial= False):
    """
    split the dataset into training and testing 
    
    """
    x,y,probe_test1= data[:,:1].flatten(),data[:,1:2].flatten(),data[:,-1:].flatten()
    
    #Saving cluster data indices in files
    if kmean:
        sampling_indices = result_folder +sp+"intial_training_indices_cluster_%s.csv"%(cluster)    
    else:
        sampling_indices = result_folder+sp+"intial_training_indices_cluster.csv" 
         
    # if indices not present in file, do random sampling and store values in particular file
    try:
        with open(sampling_indices,'r') as f:
            indices = f.read().split('\n')
            indices = [int(i) for i in indices if i]
    except Exception as e:
        indices = np.random.permutation(lt)
        np.savetxt(sampling_indices, indices, delimiter=",",fmt='%s')
     
    # split data set into training and test data set  
    training_idx, test_idx = indices[:ti], indices[ti:]
    training_idx, test_idx = indices[:ti], indices[ti:]
    Xtrain, Xtest = x[training_idx], x[test_idx]
    Ytrain, Ytest = y[training_idx], y[test_idx]
    Ptrain, Ptest = probe_test1[training_idx], probe_test1[test_idx]
    
    if radial:
        Rtest = (Xtest**2 + Ytest**2)**0.5
        Rtrain = (Xtrain**2 + Ytrain**2)**0.5
        train_ref = np.array(map(lambda x,y,r,z : [x,y,r,z] ,Xtrain,Ytrain,Rtrain,Ptrain))
        test_ref = np.array(map(lambda x,y,r : [x,y,r] ,Xtest,Ytest,Rtest))
    else:
        train_ref = np.array(map(lambda x,y,z : [x,y,z] ,Xtrain,Ytrain,Ptrain))
        test_ref = np.array(map(lambda x,y,z : [x,y,z] ,Xtest,Ytest,Ptest))
        
    return train_ref ,test_ref

def preprocessing(data,test_limit_min,test_limit_max):
    """
    remove outlier
    and zero values
    """
    zero_indices =[]
    for index,val in enumerate(data):
        if val[-1] == 0 :
            zero_indices.append(index)
            
    if zero_indices:
        data = np.delete(data,(zero_indices),axis=0)
        zero_indices = None
    
    #data[abs(data - np.mean(data)) < 3 * np.std(data)]
    #mean = np.mean(data[:,-1:])
    if data.size :
        std = np.std(data[:,-1:])
        outlier_indices = []
        for index,val in enumerate(data):
            if val[-1] > test_limit_max + abs(3*std):
                outlier_indices.append(index)
                #print "removing value outlier"
                #print val[-1]
                     
            elif (val[-1] < (test_limit_min - abs(3*std))) :
                outlier_indices.append(index)  
                #print val[-1]
                
        if outlier_indices:
            data = np.delete(data,(outlier_indices),axis=0)
        
    return data
        
def get_wafers_train_test_data(train_ref,test_ref,probe_tests,total_devices,start_index):
    
    """
    get data of each wafer corresponding to reference wafer (here first wafer)
    
    """
    s_i = start_index
    train = {}
    test = {}
    for wafer_id in total_devices.keys():
        x= np.array([float(i) for i in probe_tests[1][s_i:total_devices[wafer_id]+1]])
        y = np.array([float(i) for i in probe_tests[2][s_i:total_devices[wafer_id]+1]])
        probe_test = np.array([float(i) for i in probe_tests[pb_idx][s_i:total_devices[wafer_id]+1]])
        s_i = total_devices[wafer_id] + 1
        train[wafer_id] = []
        test[wafer_id] = []
        for x_,y_,probe_test_ in zip(x,y,probe_test):
            data_found = False
            for x1,y1,probe_test in train_ref:
                if (x_ == x1 and y_ == y1) :
                    train[wafer_id].append([x_,y_,wafer_id,probe_test_])
                    data_found = True
                    break
                
            if not data_found :
                for x1,y1,probe_test in test_ref:
                    if (x_ == x1 and y_ == y1):
                        test[wafer_id].append([x_,y_,wafer_id,probe_test_])
                        break
                
        train[wafer_id] = np.array(train[wafer_id])
        test[wafer_id] = np.array(test[wafer_id])
        
    return train, test

def get_set_of_clustered_wafers(train,test,wafer_clusters,total_devices):
    """
    group the wafers with similar trend into one cluster, and build model.
    
    """
    mean = []
    sd = []
    set_of_clustered_wafers = {}
    for wafer_id in total_devices.keys():
        mean.append(get_mean(train[wafer_id][:,-1:]))
        sd.append(np.std(train[wafer_id][:,-1:]))
    clustered_means,centroid_clusters,set_of_clustered_wafers = get_kmean_clusters_data(np.concatenate((np.array(mean).reshape(-1,1),np.array(sd).reshape(-1,1)),axis =1),clusters = wafer_clusters,return_orig_indices = True)
    for c in set_of_clustered_wafers.keys():
        c_wafers = set_of_clustered_wafers[c]
        # if cluster size more than 6, than split it into 2 
        if len(c_wafers) > 6:
            new_wafer_cluster = set_of_clustered_wafers.keys()[-1]+1
            if not new_wafer_cluster in set_of_clustered_wafers.keys():
                set_of_clustered_wafers[new_wafer_cluster] = c_wafers[:int(len(c_wafers)/2)]
                set_of_clustered_wafers[c] = c_wafers[int(len(c_wafers)/2):]
                
    return set_of_clustered_wafers
    
def get_path_style():
    
    if len(os.getcwd().split("//")) == 1:
        print "Not windows machine"
        sp = "/"
    else:
        print "windows machine"
        sp = "\\"
        
    return sp

class excel_workbook():
    """
    to save data into csv format
    
    """
    def __init__(self,headers, file_path, method_string,path_style = "\\"):
        headers = np.array(headers).reshape(1,-1)
        self.file_path = file_path
        sp = path_style
        f_n = 0
        head,tail = os.path.split(self.file_path)
        new_tail = tail.split(".")[0]+"_%s.csv"%(method_string)
        self.file_path = head +sp+ new_tail
        while (os.path.exists(self.file_path)):
            self.file_path= head +sp+ new_tail.split(".")[0]+"_%s.csv"%(f_n)
            f_n += 1
        self.open()
        self.write_data(headers)
       
    def open(self):
        self.file_handle = open(self.file_path,"ab")  
        
    def write_data(self,data):
        np.savetxt(self.file_handle, data, delimiter=",",fmt='%s')
        
    def close(self):
        self.file_handle.close()
            
    
class progressive_sampling():
    """
    Initially few samples from training budget are used to train the gaussain model. 
    More samples are progressively added from the die locations where predicted value has more standard deviation until training budget is reached.
    
    """
    def __init__(self,train,test,std,prog_sample_size):
        self.test = test
        self.train = train
        self.std = std
        self.d = spatial.distance_matrix(self.test[:,:-1], self.train[:,:-1])
        for t in range(prog_sample_size):
            max_var_id = self.prog_sampling()
            tmp_d= spatial.distance_matrix(self.test[:,:-1],self.new_training_sample[:-1].reshape(1,-1))
            self.d = np.concatenate((np.delete(self.d, (max_var_id), axis=0),tmp_d),axis=1)
        
    def prog_sampling(self):
        min_dist =[]
        for i in range(len(self.d)):
            #j, = np.where(d[i] == min(d[i]))
            min_dist.append(min(self.d[i]))
        self.std = self.std.flatten()
        min_dist = np.array(min_dist)
        sorted_dist_id = np.flip(min_dist.argsort(), axis = 0)
        max_var = -1000
        max_var_id = sorted_dist_id[-1]
        p = int(0.1 *len(sorted_dist_id)) # 20 % of min distances
        for i in sorted_dist_id[:p]:
            if max_var < self.std[i]:
                max_var = self.std[i]
                max_var_id = i
        self.new_training_sample = self.test[max_var_id]
        #plt.scatter(new_training_sample[0],new_training_sample[1],c='r')
        self.train = np.append(self.train,self.new_training_sample).reshape(-1,len(self.train[0]))
        self.test = np.delete(self.test, max_var_id, axis=0)
        self.std = np.delete(self.std, max_var_id, axis=0) 
        return max_var_id
        

    
#Read the dataset
probe_tests = pd.read_csv("probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv("test_limits.csv", engine ="python",header=None)
# Each wafer's ending index
total_devices = {0:2395,1:4784,2:7177,3:9567,4:11963,5:14357,6:16739,7:19134,8:21532,9:23925,10:26320,11:28717,12:31100,13:33493,14:35884,15:38264,
               16:40662,17:43056,18:45455,19:47847,20:50153,21:52551,22:54936}

#total_devices = {0:2395,1:4784,2:7177,3:9567,4:11963,5:14357,6:16739}
wafer_clusters = 4
min_i = 1
radial = False
kmean = True
kmean_wafer_clustering = True
#Progressive sampling steps
sampling_steps = 4
# % of training data in each sample step
#total training data budget 10%
sample_perc = 0.025  #2.5 % 

if kmean_wafer_clustering:
    method_string ="proposed_method"
else:
    method_string ="previous_method"
if kmean:
    method_string += '_cluster'
    
xref = np.array([float(i) for i in probe_tests[1][min_i:total_devices[total_devices.keys()[0]]+1]])
yref = np.array([float(i) for i in probe_tests[2][min_i:total_devices[total_devices.keys()[0]]+1]])
sp = get_path_style()

#probe test to run range 3 to 105 
for pb_idx in range(3,105):
    test_limit_min, test_limit_max = float(test_limits.loc[pb_idx-2,1]),float(test_limits.loc[pb_idx-2,2])
    
    result_folder = os.getcwd()+sp+"results"
    create_path_if_not_exists(result_folder)
    result_folder =  result_folder+sp+"%s"%(probe_tests[pb_idx][0])
    create_path_if_not_exists(result_folder)
    headers = ["X","Y",str(probe_tests[pb_idx][0])+"actual",str(probe_tests[pb_idx][0])+"predicted","stdeviation","error"]
    consolidated_path = result_folder+ sp+"consolidated_test_results.csv"
    consolidated_workbook = excel_workbook(headers,consolidated_path,method_string,sp)
    
    print "Probe_Test_Starts",pb_idx
    
    probe_test = np.array([float(i) for i in probe_tests[pb_idx][min_i:total_devices[total_devices.keys()[0]]+1]])

    if kmean:
        """
        create cluster if kmean is true
        """
        X = probe_test.reshape(-1,1)
        data_s,centroid_clusters = get_kmean_clusters_data(X,xref,yref,probe_test)
        
        """
        If values are very less in the cluster , then combine cluster data to nearest cluster
        """
        
        for ks in data_s.keys():
            if len(data_s[ks])  < 6:  # if number of data in each cluster less than 6, then combine data to nearest cluster
                val = centroid_clusters[ks]
                diff_min = 1234567
                new_cluster = ks
                for kc in data_s.keys():
                    if kc != ks:
                        diff = abs(val - centroid_clusters[kc])
                        if diff < diff_min:
                            diff_min = diff
                            new_cluster = kc   
                            
                if not (new_cluster == ks):
                    for d in data_s[ks]:
                        data_s[new_cluster].append(d)
                    del data_s[ks]
                    
                
    else:
        data_s ={0:np.array(map(lambda x,y,z : [x,y,z] ,xref,yref,probe_test))}
    
    tdx,tdy,tdz,tdp,tdsd,terror,tdpp,tdpf,tdap,tdaf,tde= np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),[],[],[],[],[]
    
    set_of_clustered_wafers ={}
    #
    for cluster in data_s.keys():
        train = {} 
        test = {}
        #cluster data---> format [[x,y,probe_test_val],]
        data = np.array(data_s[cluster])
        lt = len(data)
        #initial no. of training data
        ti = int(sample_perc*lt)
        if ti == 0:
            ti = 1
        train_ref, test_ref = train_test_split(data, ti, kmean =kmean, radial =radial)
        
        # store train and test data of each wafer in test and train dictionary
        #
        train, test = get_wafers_train_test_data(train_ref,test_ref,probe_tests,total_devices,min_i)
        
        
        if not set_of_clustered_wafers:
            if kmean_wafer_clustering:
                if len(total_devices.keys()) > 2:
                    set_of_clustered_wafers = get_set_of_clustered_wafers(train,test,wafer_clusters,total_devices)
                else:
                    set_of_clustered_wafers ={'0':[total_devices.keys()[0]],'1':[total_devices.keys()[1]]}
            else:
                set_of_clustered_wafers ={'0':total_devices.keys()}
                
        for c in set_of_clustered_wafers.keys():
            # start building model for each set of clustered wafers 
            c_wafers = set_of_clustered_wafers[c]
            training_data = []
            test_data =[]
            
            for wafer_id in c_wafers:
                test_data = np.append(test_data,test[wafer_id]).reshape(-1,len(test[wafer_id][0]))   
                training_data = np.append(training_data,train[wafer_id]).reshape(-1,len(train[wafer_id][0]))
                train[wafer_id] = None
                test[wafer_id] = None 
             
            # data preprocessing (removing outliers and zero values)   
            training_data = preprocessing(training_data, test_limit_min, test_limit_max)
            test_data =  preprocessing(test_data, test_limit_min, test_limit_max)
        
            ti = int(sample_perc * (len(training_data) + len(test_data)))
            
            if ti == 0:
                ti = 1
            
            if len(training_data)  < ti or len(training_data) < 6:
                if len(test_data)  < ti * sampling_steps + 1:
                    print "Data too less to build a model" 
                    print "Skipping"
                    continue
                else:
                    training_data  = np.append(training_data,test_data[:ti],axis =0)
                    test_data = test_data[ti:]
              
            # Hyper parameters to test  
            length_scales = np.arange(1, 50, 5)[1:]
            sigma_vals = [0,0.01,.1,.3]  # Noise values to test
            
            # Progressively take more samples from data set to train the model
            for step in range(0,sampling_steps):
                std_dev =[]
                predict =[]
                # Using self written library
                # do 5 folds cross validation to find optimal hyper parameters l and sigma 
                l_opt, sigma_opt = GP.cross_validate(training_data,l_val=length_scales,sigma_val=sigma_vals)
                # training model with optimal hyper parameters
                gp = GP.GaussianProcess(train_data=training_data,return_std = True)
                predict, std_dev = gp.predict(test_data=test_data[:,:-1], l=l_opt, sigma=sigma_opt)
                if step == sampling_steps - 1:
                    continue
                #get more training samples 
                ps = progressive_sampling(training_data,test_data,std_dev, ti)
                # new training and test data 
                training_data,test_data,std_dev = ps.train,ps.test,ps.std
                ps = None
                gp = None
            
            Ptest = test_data[:,-1:].reshape(-1,1)
            
            actual_passed = 0
            predicted_passed = 0
            escaped_test = 0
            
            # check pass and fail status of all the die
            for id in range(len(Ptest)):
                passed = False
                if (((Ptest[id][0]) > test_limit_min) 
                and ((Ptest[id][0]) < test_limit_max)):
                    actual_passed +=1 
                    passed = True
                         
                if ((predict[id][0] > test_limit_min) 
                and (predict[id][0] < test_limit_max)):
                    if passed:
                        predicted_passed +=1 
                    else:
                        escaped_test += 1
                        #low_confidence_ids.append([id,test_data[id][0],test_data[id][1]])
                '''
                if ((predict[id][0] - 1.96 * std[id][0] > test_limit_min) 
                and (predict[id][0] + 1.96 * std[id][0] < test_limit_max)):
                    if passed:
                        predicted_passed +=1 
                    else:
                        escaped_test += 1
                        low_confidence_ids.append([id,test[id][0],test[id][1]])
                '''
                        
            # save model statistics in the csv files
            headers = ["X","Y"]
            tx = np.append(test_data[:,0],["","passed" ,"failed" ,"yield_Loss","test_escapes","length_scale_opt","sigma_opt"]).reshape(-1,1)
            ty = np.append(test_data[:,1],["","" ,"" ,"","","",""]).reshape(-1,1)
            csv_data = np.concatenate((tx,ty),axis = 1)
    
            if not(actual_passed):
                actual_passed = 1
            if not(predicted_passed):
                predicted_passed = 1 # for div by zero 
                         
            actual_failed = len(predict) - actual_passed  #len(predict) total no of test dies
            predicted_failed = len(predict) - predicted_passed - escaped_test  
            yield_loss_actual = float(actual_failed)*100/(actual_passed+actual_failed)
            yield_loss_predicted = float(predicted_failed)*100/(predicted_passed+predicted_failed)
            consolidated_workbook.write_data(np.concatenate((test_data[:,0:1],test_data[:,1:2],Ptest,predict,std_dev,abs(Ptest-predict)*100/ Ptest),axis=1))
            
            tdpp.append(predicted_passed),tdpf.append(predicted_failed),tdap.append(actual_passed),tdaf.append(actual_failed),tde.append(escaped_test)
            Ptest =  np.append(Ptest,["",actual_passed ,actual_failed ,yield_loss_actual ,"","",""]).reshape(-1,1) 
            predict =  np.append(predict,["",predicted_passed ,predicted_failed ,yield_loss_predicted,escaped_test,l_opt,sigma_opt]).reshape(-1,1)
            std_dev =  np.append(std_dev,["","" ,"" ,"","","",""]).reshape(-1,1)
            
            if len(Ptest) - len(training_data) < 0:
                Ptest = np.append(Ptest,[""]* (len(training_data)-len(Ptest))).reshape(-1,1)
                predict = np.append(predict,[""]* (len(training_data)-len(predict))).reshape(-1,1)
                std_dev = np.append(std_dev,[""]* (len(training_data)-len(std_dev))).reshape(-1,1)
                csv_data = np.append(csv_data,[""]* (len(csv_data[0])*(len(training_data)-len(csv_data)))).reshape(-1,len(csv_data[0]))
                
            tr_x = np.append(training_data[:,0],[""]*(len(Ptest)-len(training_data))).reshape(-1,1)
            tr_y = np.append(training_data[:,1],[""]*(len(Ptest)-len(training_data))).reshape(-1,1)
            tr_pb = np.append(training_data[:,-1],[""]*(len(Ptest)-len(training_data))).reshape(-1,1)
            
            headers = ['Training_data_X','Training_data_Y','Training_data_%s'%str(probe_tests[pb_idx][0])]+headers+[str(probe_tests[pb_idx][0])+'_actual_cluster_'+str(cluster),str(probe_tests[pb_idx][0])+'_predicted_cluster_'+str(cluster),'std_dev_'+str(cluster)]
            
            if kmean:
                ms =method_string+ '_%s'%(cluster) +"_wafers_"+ "_".join([str(f) for f in c_wafers])
            else:
                ms = method_string+"_wafers_"+"_".join([str(f) for f in c_wafers])
            csv_path = result_folder + sp+"test_results.csv"
            workbook = excel_workbook(headers,csv_path,ms,sp)
            workbook.write_data(np.concatenate((tr_x,tr_y,tr_pb,csv_data, Ptest, predict,std_dev),axis=1))
            workbook.close()
            workbook = None
            Ptest,predict,std_dev,tr_x,tr_y,tr_z,test_data,training_data = None,None,None,None,None,None,None,None
            print "completed cluster %s predictions of Test %s"%(cluster,pb_idx)
     
    
    tdx = np.array(["","passed" ,"failed" ,"yield_Loss","test_escapes"]).reshape(-1,1)
    tdy = np.array(["","" ,"" ,"",""]).reshape(-1,1)
    tdz =  np.array(["",sum(tdap) ,sum(tdaf) ,float(sum(tdaf))*100/(sum(tdap)+sum(tdaf)) ,""]).reshape(-1,1) 
    tdp =  np.array(["",sum(tdpp) ,sum(tdpf) ,float(sum(tdpf))*100/(sum(tdpp)+sum(tdpf)),sum(tde)]).reshape(-1,1)
    consolidated_workbook.write_data(np.concatenate((tdx,tdy,tdz,tdp,tdy,tdy),axis=1))
    consolidated_workbook.close()
    consolidated_workbook = None
    print "completed probe test %s"%pb_idx
  
