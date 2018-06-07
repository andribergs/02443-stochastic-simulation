from math import log, floor, factorial, sqrt
import statistics as st
import random as random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import stats_utils as utils
    

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
                "erlang": lambda: stats.erlang.rvs(2, size=n_customers, scale=mtbc),
                #p_1 = 0.8, lambda_1 = 0.8333, p_2 = 0.2, lambda_2 = 5.0
                "hyper_exp": 1+1
            }
    service_time_dist_types = {
                "exp": lambda: stats.expon.rvs(size=n_customers, scale=mst)
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
    
def event_simulation_erlang_arrival(n_su, mst, mtbc, n_sims, n_customers):  
    event_simulation(n_su, mst, mtbc, n_sims, n_customers, "erlang", "exp")
    
def event_simulation_hyper_exponential_arrival(n_su, mst, mtbc, n_sims, n_customers):
    event_simulation(n_su, mst, mtbc, n_sims, n_customers, "hyper_exp", "exp")
    
def calculate_confidence_intervals(mean, standard_deviation, n_simulations):
    z_s = stats.t.ppf(0.95, n_simulations)
    lower = mean - z_s * (standard_deviation/sqrt(n_simulations))
    upper = mean + z_s * (standard_deviation/sqrt(n_simulations))
    return (lower, upper)
        
def erlang_B_formula(n, arrival_intensity, mean_service_time):
    A = arrival_intensity * mean_service_time
    B = ((A**n) / (factorial(n))) / (sum([((A**i)/factorial(i)) for i in range(n)]))
    return B  
    

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
    
    #Simulation with a renewal process using Erlang distributed inter arrival times
    event_simulation_erlang_arrival(n_service_units, mean_service_time, mean_time_between_customers, n_simulations, n_customers)
    
    #Simulation with a renewal process using hyper exponential inter arrival times
    #event_simulation_hyper_exponential_arrival(n_service_units, mean_service_time, mean_time_between_customers, n_simulations, n_customers)
    
    #Exact solution
    print("Exact solution, percentage of blocked customers: {} %".format(erlang_B_formula(n_service_units, mean_time_between_customers, mean_service_time)* 100))
if __name__ == "__main__":
    main()
