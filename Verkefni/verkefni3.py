#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:56:27 2018

@author: thorsteinngj
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, norm, expon, t

#%%
#---- Exponential Distribution ----
"""
Generate simulated values from the following distributions
⋄ Exponential distribution
• Verify the results by comparing histograms with analytical
results and erform tests for distribution type.
"""
def expo(lam,U):
    res = (-np.log(U)/lam)
    return res


#Rate parameter
U = np.random.uniform(0.0,1.0,10000)
lam = 1
res = expo(lam,U)
x = np.linspace(expon.ppf(0),expon.ppf(0.99999),100)


#Histogram
plt.figure()
plt.hist(res,align='mid',color='tan',edgecolor='moccasin',bins=10,density=True,stacked=True)
xmin, xmax = plt.xlim()
plt.plot(x, expon.pdf(x),'g-', lw=2,alpha=0.6)
plt.title("Exponentially Distributed Histogram")
plt.xlabel("Classes")
plt.ylabel("Density")
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
mud, stdd = norm.fit(res2)
#Histogram
plt.figure()
plt.hist(res2,color='tan',align='mid',edgecolor='moccasin',bins=10,density=True,stacked=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mud, stdd)
plt.plot(x, p,'g-', lw=2,alpha=0.6)
plt.title("Normal Distributed Histogram")
plt.xlabel("Classes")
plt.ylabel("Density")
plt.show

#%% 
# ----- Confidence intervals -----
ulim = []
llim = []
n = 10 # 10 observations
m = 100 # 100 Number of intervals
for i in range(m):
    x = normo(0,1,n)
    x_bar = np.mean(x)
    z_star = t.ppf(0.975,n-1)
    upper = x_bar + z_star*(np.std(x)/np.sqrt(n))
    lower = x_bar - z_star*(np.std(x)/np.sqrt(n))
    ulim = np.append(ulim,upper)
    llim = np.append(llim,lower)
    

xlim = [np.min(llim),np.max(ulim)]

# plot the upper and lower limit
plt.figure()
plt.plot(ulim,np.zeros(m),'r>',alpha=0.5, label = "Upper")
plt.plot(llim,np.zeros(m),'b<',alpha=0.5, label = "Lower")
xmin, xmax = plt.xlim()
plt.xlabel('Confidence intervals')
plt.legend()
plt.show()

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
    mean = (k/(k-1))*beta-1
    variance = (k/((k-1)**2*(k-2)))*beta**2
    return res, mean, variance

n = 10000
beta = 1
k1 = 2.05; k2 = 2.5; k3 = 3; k4 = 4

#First k
res31, mean1, var1 = paretobay(beta,k1,n)
anamean1 = np.mean(res31)
anavar1 = np.var(res31)

x1 = np.linspace(pareto.ppf(0.01, k1),pareto.ppf(0.9999,k1),100)

plt.figure()
plt.hist(res31,align='mid',color='tan',edgecolor='moccasin',bins=20,density=True,stacked=True)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(x1-1, pareto.pdf(x1, k1),'g-', lw=2,alpha=0.6)
plt.ylim(ymin,ymax)
plt.title("Pareto Distributed Histogram (k=2.05)")
plt.xlabel("Classes")
plt.ylabel("Density")
plt.show
print('----Pareto with K = 2.05----')
print('The theoretical mean is: {0}'.format(mean1))
print('The theoretical variance is: {0}'.format(var1))
print('The analytical mean is: {0}'.format(anamean1))
print('The analytical variance is: {0}'.format(anavar1))

#Second k
res32, mean2, var2 = paretobay(beta,k2,n)
anamean2 = np.mean(res32)
anavar2 = np.var(res32)

x2 = np.linspace(pareto.ppf(0.01, k2),pareto.ppf(0.9999,k2),100)

plt.figure()
plt.hist(res32,align='mid',color='tan',edgecolor='moccasin',bins=20,density=True,stacked=True)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(x2-1, pareto.pdf(x2, k2),'g-', lw=2,alpha=0.6)
plt.ylim(ymin,ymax)
plt.title("Pareto Distributed Histogram (k=2.5)")
plt.xlabel("Classes")
plt.ylabel("Density")
plt.show
print('----Pareto with K = 2.5----')
print('The theoretical mean is: {0}'.format(mean2))
print('The theoretical variance is: {0}'.format(var2))
print('The analytical mean is: {0}'.format(anamean2))
print('The analytical variance is: {0}'.format(anavar2))

#Third k
res33, mean3, var3 = paretobay(beta,k3,n)
anamean3 = np.mean(res33)
anavar3 = np.var(res33)

x3 = np.linspace(pareto.ppf(0.01, k3),pareto.ppf(0.9999,k3),100)

plt.figure()
plt.hist(res33,align='mid',color='tan',edgecolor='moccasin',bins=20,density=True,stacked=True)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(x3-1, pareto.pdf(x3, k3),'g-', lw=2,alpha=0.6)
plt.ylim(ymin,ymax)
plt.title("Pareto Distributed Histogram (k=3)")
plt.xlabel("Classes")
plt.ylabel("Density")
plt.show
print('----Pareto with K = 3----')
print('The theoretical mean is: {0}'.format(mean3))
print('The theoretical variance is: {0}'.format(var3))
print('The analytical mean is: {0}'.format(anamean3))
print('The analytical variance is: {0}'.format(anavar3))



#Fourth k
res34, mean4, var4 = paretobay(beta,k4,n)
anamean4 = np.mean(res34)
anavar4 = np.var(res34)

x4 = np.linspace(pareto.ppf(0.01, k4),pareto.ppf(0.9999,k4),100)

plt.figure()
plt.hist(res34,align='mid',color='tan',edgecolor='moccasin',bins=20,density=True,stacked=True)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(x4-1, pareto.pdf(x4, k4),'g-', lw=2,alpha=0.6)
plt.ylim(ymin,ymax)
plt.title("Pareto Distributed Histogram (k=4)")
plt.xlabel("Classes")
plt.ylabel("Density")
plt.show
print('----Pareto with K = 4----')
print('The theoretical mean is: {0}'.format(mean4))
print('The theoretical variance is: {0}'.format(var4))
print('The analytical mean is: {0}'.format(anamean4))
print('The analytical variance is: {0}'.format(anavar4))


    
