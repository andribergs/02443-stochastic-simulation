from math import sqrt, log, floor
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import stats_utils as utils


def geometric_distribution(rn, p):
    return [w + 1 for w in [floor(z) for z in [ y/log(1-p) for y in [log(x) for x in rn]]]]

def six_point_crude_method(rn):
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    a = 1+1
    
def six_point_rejection_method(rn):
    a = 1+1
    
def six_point_alias_method():
    a = 1+1

def main():
    #Generate 10000 pseudo-random numbers
    rn = list(np.random.random_sample(10000))
    
    #Choose a value for p in the geometric distribution and simulate 10,000 outcomes.
    p = 0.5
    X = geometric_distribution(rn, p)
    Y = stats.geom.rvs(p, size=10000)
    N, bins = utils.histogram(X, "Simulated geometric distribution", max(X))
    
    n_observed = utils.sort_values_to_bins(X)
    n_expected = utils.sort_values_to_bins(Y)
    test_stat, p_chi = utils.chi_squared_test(n_observed, n_expected)
    print("Chi squared test")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    
    #Simulate 6 point distribution using crude method
    six_point_crude_method(rn)
        

if __name__ == "__main__":
    main()
