#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:56:27 2018

@author: thorsteinngj
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto


#---- Exponential Distribution ----
"""
Generate simulated values from the following distributions
⋄ Exponential distribution
• Verify the results by comparing histograms with analytical
results and erform tests for distribution type.
"""
def expo(lam):
    U = np.random.uniform(0.0,1.0,10000)
    res = (-np.log(U)/lam)
    return res

#Rate parameter
lam = 1
res = expo(lam)
u = np.exp(U)

#Histogram
#Vantar að setja samanburðarlinu.
plt.hist(res,align='mid',color='tan',edgecolor='moccasin',bins=100)
plt.title("Exponentially Distributed Histogram")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show
#%%
#---- Normal Distribution ----
"""
Generate simulated values from the following distributions
⋄ Normal distribution (at least with standard Box-Mueller)
• Verify the results by comparing histograms with analytical
results and erform tests for distribution type.
For the normal distribution generate 100 95% confidence
intervals for the mean and variance based on 10 observations.
Discuss the results.
"""
def normo(mu,sigma,n):    
    G = []
    for i in range(int(n/2)):
        U1 = np.random.uniform(0,1,1)
        U2 = np.random.uniform(0,1,1)
        phi = 2*np.pi*U2
        r = np.sqrt(-2*np.log(U1))
        sine = np.sin(phi)
        cose = np.cos(phi)
        together = np.array([cose, sine])
        G = np.append(G,r*together)
    print(G)
    res = mu + sigma*G
    return res

mu = 0
sigma = 1
n = 10000
res2 = normo(mu,sigma,n)
#Histogram
#Vantar að setja samanburðarlinu.
plt.hist(res2,color='tan',align='mid',edgecolor='moccasin',bins=100)
plt.title("Normal Distributed Histogram")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show

#%% 
# ----- Confidence intervals -----
ulim = []
llim = []
ulim1 = ulim
llim1 = llim
n = 10 # 10 observations
m = 100 # 100 Number of intervals
for i in range(m):
    x = normo(0,1,n)
    x_bar = np.mean(x)
    z_star = stats.t.ppf(0.975,n-1)
    #Vil eg nota einn eða s?
    upper = x_bar + z_star*(np.std(x)/np.sqrt(n))
    lower = x_bar - z_star*(np.std(x)/np.sqrt(n))
    ulim = np.append(ulim,upper)
    llim = np.append(llim,lower)
    upper1 = x_bar + 1.96/np.sqrt(n)
    lower1 = x_bar - 1.96/np.sqrt(n)
    ulim1 = np.append(ulim1,upper1)
    llim1 = np.append(llim1,lower1)

xlim = [np.min(llim),np.max(ulim)]
xlim1 = [np.min(llim1),np.max(ulim1)]

# plot the upper and lower limit
plt.figure()
plt.plot(ulim,np.zeros(m),'r>')
plt.plot(llim,np.zeros(m),'b<')
plt.xlim(-2,2)
plt.xlabel('Confidence intervals')
plt.legend('UL')

#%%
#---- Pareto Distribution ----
"""
Generate simulated values from the following distributions
⋄ Pareto distribution, with β = 1 and experiment with different
values of k values: k = 2.05, k = 2.5, k = 3 og k = 4.
• Verify the results by comparing histograms with analytical
results and perform tests for distribution type!!!
• For the Pareto distribution with support on [β,∞[ compare
mean value and variance, with analytical results.
"""
def paretobay(beta,k,n):
    U = np.random.uniform(0,1,n)
    res = beta*(U**(-1/k)-1)
    mean = (k/(k-1))*beta
    variance = (k/((k-1)**2*(k-2)))*beta**2
    return res, mean, variance

n = 10000
beta = 1
k1 = 2.05; k2 = 2.5; k3 = 3; k4 = 4

#Fyrsta dæmi
res31, mean1, var1 = paretobay(beta,k1,n)
anamean1 = np.mean(res31)
anavar1 = np.var(res31)
plt.figure()
plt.hist(res31,align='mid',color='tan',edgecolor='moccasin',bins=100)
plt.title("Pareto Distributed Histogram (k=2.05)")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show
print('----Pareto with K = 2.05----')
print('The theoretical mean is: {0}'.format(mean1))
print('The theoretical variance is: {0}'.format(var1))
print('The analytical mean is: {0}'.format(anamean1))
print('The analytical variance is: {0}'.format(anavar1))

#Annað dæmi
res32, mean2, var2 = paretobay(beta,k2,n)
anamean2 = np.mean(res32)
anavar2 = np.var(res32)
plt.figure()
plt.hist(res32,align='mid',color='tan',edgecolor='moccasin',bins=100)
plt.title("Pareto Distributed Histogram (k=2.5)")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show
print('----Pareto with K = 2.5----')
print('The theoretical mean is: {0}'.format(mean2))
print('The theoretical variance is: {0}'.format(var2))
print('The analytical mean is: {0}'.format(anamean2))
print('The analytical variance is: {0}'.format(anavar2))

#Þriðja dæmi
res33, mean3, var3 = paretobay(beta,k3,n)
anamean3 = np.mean(res33)
anavar3 = np.var(res33)
plt.figure()
plt.hist(res33,align='mid',color='tan',edgecolor='moccasin',bins=100)
plt.title("Pareto Distributed Histogram (k=3)")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show
print('----Pareto with K = 3----')
print('The theoretical mean is: {0}'.format(mean3))
print('The theoretical variance is: {0}'.format(var3))
print('The analytical mean is: {0}'.format(anamean3))
print('The analytical variance is: {0}'.format(anavar3))



#Fyrsta dæmi
res34, mean4, var4 = paretobay(beta,k4,n)
anamean4 = np.mean(res34)
anavar4 = np.var(res34)
plt.figure()
plt.hist(res34,align='mid',color='tan',edgecolor='moccasin',bins=100)
plt.title("Pareto Distributed Histogram (k=4)")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show
print('----Pareto with K = 4----')
print('The theoretical mean is: {0}'.format(mean4))
print('The theoretical variance is: {0}'.format(var4))
print('The analytical mean is: {0}'.format(anamean4))
print('The analytical variance is: {0}'.format(anavar4))


    
