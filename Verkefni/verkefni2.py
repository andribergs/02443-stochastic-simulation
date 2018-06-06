#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:56:27 2018

@author: thorsteinngj
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

"""
Generate simulated values from the following distributions
⋄ Exponential distribution
⋄ Normal distribution (at least with standard Box-Mueller)
⋄ Pareto distribution, with β = 1 and experiment with different
values of k values: k = 2.05, k = 2.5, k = 3 og k = 4.
• Verify the results by comparing histograms with analytical
results and erform tests for distribution type.
• For the Pareto distribution with support on [β,∞[ compare
mean value and variance, with analytical results, which can be
calculated as E{X} = β
k
k−1
(for k > 1) and
V ar{X} = β
2 k
(k−1)2(k−2) (for k > 2)
For the normal distribution generate 100 95% confidence
intervals for the mean and variance based on 10 observations.
Discuss the results.
"""

#---- Exponential Distribution ----
def expo(U,lam):
    res = (-np.log(U)/lam)
    return res

U = np.random.uniform(0.0,1.0,10000)
#Rate parameter
lam = 1
res = expo(U,lam)
u = np.exp(U)

#Histogram
#Vantar að setja samanburðarlinu.
plt.hist(res,align='mid',color='tan',edgecolor='moccasin')
plt.title("Exponential Histogram")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show

#---- Normal Distribution ----
def normo(mu,sigma,n):    
    Un = np.random.uniform(0.0,1.0,1)
    G = []
    for i in range(n/2):
        sine = np.sin(2*pi*Un)
        cose = np.cos(2*pi*Un)
        together = np.array([sine, cose])
        G = [G, np.sqrt(-2*np.log(Un)*[sin_part, cos_part])] 
        res = mu+sigma*G
    return res
    