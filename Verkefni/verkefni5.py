#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:17:15 2018

@author: thorsteinngj
"""

import numpy as np
from scipy import stats

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
z_star = stats.t.ppf(0.975,n-1)
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
z_star = stats.t.ppf(0.975,n-1)
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
z_star = stats.t.ppf(0.975,n-1)
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
def queue_simulation(n_su, mst, mtbc, n_customers, arrival_dist, service_time_dist):
    blocked_customer_count = 0
    time = list(np.cumsum(arrival_dist))
    service_unit_times = [0 for _ in range(n_su)]

    for i in range(n_customers):
        min_service_unit_time = min(service_unit_times)
        if time[i] > min_service_unit_time:
            service_unit_time = service_time_dist[i] + time[i]
            service_unit_times[service_unit_times.index(min_service_unit_time)] = service_unit_time
        else:
            blocked_customer_count = blocked_customer_count + 1

    return blocked_customer_count

def event_simulation(n_su, mst, mtbc, n_sims, n_customers, arr_type, serv_type): 
    results = []
    arrival_dist_types = {
                #interarrival intervals are exponentionally distributed when arrival process is a Poisson process
                "poisson": lambda: stats.expon.rvs(size=n_customers, scale=mtbc),
                #p_1 = 0.8, lambda_1 = 0.8333, p_2 = 0.2, lambda_2 = 5.0
                "hyper_exp": lambda: [stats.expon.rvs(scale=1/0.8333) if u < 0.8 else stats.expon.rvs(scale=1/5) for u in list(np.random.random_sample(10000))]
            }
    service_time_dist_types = {
                "exp": lambda: stats.expon.rvs(size=n_customers, scale=mst),
                "constant": lambda: [mst for _ in range(n_customers)],
                #k=1.05 or k=2.05 will make interesting choices
                "pareto": lambda: [((mst*(2.05-1))/2.05) / (np.random.uniform(0,1)**(1/2.05)) for _ in range(n_customers)]
            }
    
    for i in range(n_sims):
        arrival_dist = arrival_dist_types[arr_type]()
        service_time_dist = service_time_dist_types[serv_type]()
        blocked_customer_count = queue_simulation(n_su, mst, mtbc, n_customers, arrival_dist, service_time_dist)
        results.append(blocked_customer_count / n_customers)
        
    blocked_customers_fraction = sum(results) / n_sims
    confidence_intervals = calculate_confidence_intervals(st.mean(results), st.stdev(results), n_sims)
    
    print("Simulation with the arrival process as {} and the service time distribution as {}".format(arr_type, serv_type))
    print("-------------------------------------------------------------------------")
    print("Percentage of blocked customers: {} %".format(blocked_customers_fraction * 100))
    print("Confidence interval, lower limit: {} %".format(confidence_intervals[0] * 100))
    print("Confidence interval, upper limit: {} %".format(confidence_intervals[1] * 100))
    print("-------------------------------------------------------------------------")
    print("\n")

def event_simulation_poisson_arrival(n_su, mst, mtbc, n_sims, n_customers):    
    event_simulation(n_su, mst, mtbc, n_sims, n_customers, "poisson", "exp")
    
def main():
    #service units = 10, 
    #mean service time = 8 time units, 
    #mean timebetween customers = 1 time unit
    #10 x 10.000 customers.
    n_service_units = 10
    mean_service_time = 8
    mean_time_between_customers = 1
    n_simulations = 10
    n_customers = 10000
    
    #Simulation for when the arrival process is modelled as a Poisson process
    event_simulation_poisson_arrival(n_service_units, mean_service_time, mean_time_between_customers, n_simulations, n_customers)
    
        #Exact solution
    print("Exact solution, percentage of blocked customers: {} %".format(erlang_B_formula(n_service_units, mean_time_between_customers, mean_service_time)* 100))
if __name__ == "__main__":
    main()
