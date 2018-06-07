#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:17:15 2018

@author: thorsteinngj
"""

import numpy as np
import scipy.stats as st

#%%
#• Estimate the integral exdx (from 0 to 1) by simulation (the crude Monte
#Carlo estimator). Use eg. an estimator based on 100 samples
#and present the result as the point estimator and a confidence
#interval.

#Theoretical variance: 0.2420
n = 100
U = np.random.uniform(0,1,n)
X = np.exp(U)
X_bar_c = np.sum(X)/n #mean
Var_c = np.sum(X**2)/n - (np.sum(X)/n)**2
theo_c = 0.2420
estimate = 1.7183

#Confidence interval
z_star = st.t.ppf(0.975,n-1)
upper_c = X_bar_c + z_star*(np.std(X)/np.sqrt(n))
lower_c = X_bar_c - z_star*(np.std(X)/np.sqrt(n))

print('----Crude Monte Carlo Evaluator----')
print("The point estimator is: {0}".format(X_bar_c))
print("The correct estimate is: {0}".format(estimate))
print("The variance is: {0}".format(Var_c))
print("The theoretical variance is: {0}".format(theo_c))
print("The upper confidence interval with 95% certainty is: {}".format(upper_c))
print("The lower confidence interval with 95% certainty is: {}".format(lower_c))
#%%
#• Estimate the integral exdx (from 0 to 1) using antithetic variables, with
#comparable computer ressources.
Y = (np.exp(U) + np.exp(1-U))/2
Y_bar_a = np.sum(Y)/n
Var_a = np.sum(Y**2)/n - (np.sum(Y)/n)**2
theo_a = 0.0039
#v = np.cov(np.exp(U))

#Confidence interval
z_star = st.t.ppf(0.975,n-1)
upper_a = Y_bar_a + z_star*(np.std(Y)/np.sqrt(n))
lower_a = Y_bar_a - z_star*(np.std(Y)/np.sqrt(n))

print('----Antithetic Monte Carlo Evaluator----')
print("The point estimator is: {0}".format(Y_bar_a))
print("The correct estimate is: {0}".format(estimate))
print("The variance is: {0}".format(Var_a))
print("The theoretical variance is: {0}".format(theo_a))
print("The upper confidence interval with 95% certainty is: {}".format(upper_a))
print("The lower confidence interval with 95% certainty is: {}".format(lower_a))

#%%
#• Estimate the integral exdx (from 0 to 1) using a control variable, with
#comparable computer ressources.
X = np.exp(U)
Y = U
meanY = np.mean(Y)
VarY = np.sum(Y**2)/n - (np.sum(Y)/n)**2
CovY = np.sum(X*Y.T)/n-(np.sum(X)/n)*np.sum(Y.T)/n
c = -CovY/VarY
Z = X + c*(Y-meanY)
Z_bar = np.sum(Z)/n
Var_c = np.sum(Z**2)/n - (np.sum(Z)/n)**2
theo_c = 0.0039

#Confidence interval
z_star = st.t.ppf(0.975,n-1)
upper_c = Z_bar + z_star*(np.std(Z)/np.sqrt(n))
lower_c = Z_bar - z_star*(np.std(Z)/np.sqrt(n))

print('----Antithetic Monte Carlo Evaluator----')
print("The point estimator is: {0}".format(Z_bar))
print("The correct estimate is: {0}".format(estimate))
print("The variance is: {0}".format(Var_c))
print("The theoretical variance is: {0}".format(theo_c))
print("The upper confidence interval with 95% certainty is: {}".format(upper_c))
print("The lower confidence interval with 95% certainty is: {}".format(lower_c))

#%%
#• Estimate the integral exdx (from 0 to 1) using stratified sampling, with
#comparable computer ressources.

W = 0
strata = 10
#Kannski skipta upp U-inu? Kemur eitthvað skrytið ut
for j in range(strata):
    W = W + np.exp((j-1)/strata + U/strata)
                   
W = W/strata
W_bar = np.sum(W)/n
Var_s = np.sum(W**2)/n - (np.sum(W)/n)**2

#Confidence interval
z_star = st.t.ppf(0.975,n-1)
upper_w = W_bar + z_star*(np.std(W)/np.sqrt(n))
lower_w = W_bar - z_star*(np.std(W)/np.sqrt(n))

print('----Antithetic Monte Carlo Evaluator----')
print("The point estimator is: {0}".format(W_bar))
print("The correct estimate is: {0}".format(estimate))
print("The variance is: {0}".format(Var_s))
#print("The theoretical variance is: {0}".format(theo_c))
print("The upper confidence interval with 95% certainty is: {}".format(upper_w))
print("The lower confidence interval with 95% certainty is: {}".format(lower_w))

#%%
#• Use control variates to reduce the variance of the estimator in
#exercise 4 (Poisson arrivals).