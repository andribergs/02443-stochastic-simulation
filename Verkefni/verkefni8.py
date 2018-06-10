#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:38:11 2018

@author: thorsteinngj
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#Let X1-Xn be independent identically distributed random variables with unknown
#mean. For given constant a < b, wanna estimate p = P(a less than sum Si/n - mean less than b)

#a) How to use bootstrap to estimate p.
#b) Estimate p if n = 10 and values of X_i are ..., a = -5, b = 5
n = 10
a = -5
b = 5
X = np.array([56,101,78,67,93,87,64,72,80,69])

r = 100 #Bootstrap value?
mean_res = []
for i in range(r):
    #Making the bootstrap samples
    samples = np.random.choice(X,10)
    
    #Calculate the mean of samples
    mean_diff = np.mean(samples) - np.mean(X)
    mean_res = np.append(mean_res,mean_diff)
#Ratio of p values within a and b
p = np.sum(np.abs(mean_res)<5)/1

print('Estimated p is {}%'.format(p))
    

#%%

def bootcalc(distribution,r):
    median_res = []
    mean_res = []
    for i in range(r):
        samples = np.random.choice(distribution,np.size(distribution))
        median_res = np.append(median_res,np.median(samples))
        mean_res = np.append(mean_res,np.mean(samples))
        
    #Variance
    var_res_mean = np.sum((mean_res - np.mean(distribution))**2)/ (np.size(mean_res)-1)
    var_res_median = np.sum((median_res - np.median(distribution))**2)/(np.size(median_res)-1)
    return median_res, mean_res, var_res_mean, var_res_median

X = np.array([56,101,78,67,93,87,64,72,80,69])
r = 100
med,mean,varmean,varmed = bootcalc(X,r)
print("------Exercise 13------")
print('The bootstrapped estimate of the variance of the sample mean is: {0}'.format(varmean))
print('The bootstrapped estimate of the vatiance of the sample median is: {0}'.format(varmed))

def paretobay(beta,k,n):
    U = np.random.uniform(0,1,n)
    res = beta*(U**(-1/k)-1)
    mean = (k/(k-1))*beta-1
    variance = (k/((k-1)**2*(k-2)))*beta**2
    return res, mean, variance

n = 200
beta = 1
k = 1.05

res, mean, variance = paretobay(beta,k,n)
med = np.median(res)

medp,meanp,varmeanp,varmedp = bootcalc(res,r)
print("------Pareto------")
print('The bootstrapped estimate of the variance of the sample mean is: {0}'.format(varmeanp))
print('The bootstrapped estimate of the vatiance of the sample median is: {0}'.format(varmedp))
