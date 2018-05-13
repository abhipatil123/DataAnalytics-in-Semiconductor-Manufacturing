'''

Created on Mar 9, 2018

@author: sps

'''
from scipy import spatial
import numpy as np


class GaussianProcess:
    """
    Gaussian Process class for predicting probe test values
    
    Input to model  : 2 dimensional wafer coordinates (x,y) 
    Output : Predicted probe test value
    
    """

    def __init__(self, train_data, return_std = False):
        self.XY = train_data[:, :-1]
        self.Z = train_data[:, -1:]
        self.return_std = return_std

    def predict(self, test_data, l, sigma):
        K_xtest_x = GaussianKernel(test_data, self.XY, l)
        K_xtest_xtest = GaussianKernel(test_data, test_data, l)
        K = GaussianKernel(self.XY, self.XY, l)
        inv = np.linalg.inv(K + sigma**2 * np.eye(len(self.XY)))
        predictions = K_xtest_x.dot(inv).dot(self.Z)
        var = np.diag(K_xtest_xtest- K_xtest_x.dot(inv).dot(K_xtest_x.T))
        var.setflags(write=1)
        var_negative = var < 0
        if np.any(var_negative):
            var[var_negative] = 0.0
        std =  ((var)**0.5).reshape(-1,1)

        if self.return_std:
            return predictions,std
        else:
            return predictions


def squared_spatial_matrix(xcord,ycord):
    
    """
    Squared spatial distance matrix
    
    """
    return (spatial.distance_matrix(xcord, ycord))**2

def squared_exponential_kernel(squared_spatial_distance_matrix,l):
    """
    Gaussian kernel
    
    """
    return np.exp(-(squared_spatial_distance_matrix) / (2 * l * l))

def GaussianKernel(xcord, ycord, l):
    """
    Squared exponential covariance function.
    """
    d2 = squared_spatial_matrix(xcord, ycord)
    k = squared_exponential_kernel(d2,l)
    return k
    

def cross_validate (train_data, l_val, sigma_val, k_folds = 5):
    """
    K fold cross validation for the Gaussian process model.
    
    """
    rmse_opt = 1000000
    
    d2_xtest_x = {}  
    d2  = {}  
    X = {}
    Y = {}
    test_y ={}

    for k in xrange(k_folds):
        folds = np.array_split(train_data, k_folds)
        test_data = folds.pop(k)
        train_data = np.concatenate(folds)
        X[k] = train_data[:, :-1]
        Y[k] = train_data[:, -1:]
        test_y[k] = test_data[:, -1:]
        d2_xtest_x[k] = squared_spatial_matrix(test_data[:, :-1], X[k])
        d2[k] = squared_spatial_matrix(X[k], X[k])
        
    rmse_pre = rmse_opt
    l_opt = 49
    sigma_opt = 0.01
    for l in l_val:
        for sigma in sigma_val:
            for k in xrange(k_folds):
                try:
                    inv = np.linalg.inv(squared_exponential_kernel(d2[k], l) + sigma**2 * np.eye(len(X[k])))
                    pred = squared_exponential_kernel(d2_xtest_x[k], l).dot(inv).dot(Y[k])
                    rmse = (((pred - test_y[k])**2).mean())**.5     #root mean squared error
                except Exception as e:
                    rmse = 10000
                    
                if rmse < rmse_opt:
                    rmse_opt = rmse
                    l_opt = l
                    sigma_opt = sigma
        
        if not(rmse_opt < rmse_pre) :
            break
        else:
            rmse_pre = rmse_opt

    return l_opt, sigma_opt


