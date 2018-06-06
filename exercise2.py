from math import sqrt, log, floor
import numpy as np
import scipy as sp
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
    X = geometric_distribution(rn, 0.5)
    N, bins = utils.histogram(X, "Simulated geometric distribution", max(X))
    
    
    #Simulate 6 point distribution using crude method
    six_point_crude_method(rn)
        

if __name__ == "__main__":
    main()
