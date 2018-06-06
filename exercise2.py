from math import log, floor
from random import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import stats_utils as utils


def geometric_distribution(U, p):
    return [w + 1 for w in [floor(z) for z in [ y/log(1-p) for y in [log(x) for x in U]]]]


def histogram_comparison_2d(x, y, title="Histogram", n_bins=10):
    plt.hist([x,y], n_bins, alpha=0.5, label=["Simulated","Expected"], color=["blue", "red"])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel("Number of bins: {0}".format(n_bins))
    plt.ylabel("Amount in each bin")
    plt.show()


def six_point_crude_method(U):
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    x_i = [1, 2, 3, 4, 5, 6]
    n = len(U)
    X = []
    for i in range(n):
        if U[i] > max(p_i):
            x_value = x_i[p_i.index(max(p_i))]
        else:
            x_value = x_i[p_i.index(min([x for x in p_i if x >= U[i]]))]
        X.append(x_value)
    return X
    
def six_point_rejection_method():
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    q_i = [1/6 for _ in range(6)]
    C = 2
    n = len(p_i)
    X = []
    while len(X) < 10000:
        y = 1 + floor(n * random.uniform(0,1))
        u2 = random.uniform(0,1)
        if u2 < (p_i[y]/C*q_i[j]):
            X.append(y)
    return X
        
    
def six_point_alias_method():
    a = 1+1

def main():
    #Generate 10000 pseudo-random numbers
    U = list(np.random.random_sample(10000))
    
    #Choose a value for p in the geometric distribution and simulate 10,000 outcomes.
    p = 0.5
    X = geometric_distribution(U, p)
    Y = stats.geom.rvs(p, size=10000)
    histogram_comparison_2d(X, Y, "Geometric Distribution", max(max(X), max(Y)) - 1)
    print(max(max(X), max(Y)))

    n_observed = utils.sort_values_to_bins(X)
    n_expected = utils.sort_values_to_bins(Y)
    test_stat, p_chi = utils.chi_squared_test(n_observed, n_expected)
    print("Chi squared test")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    
    #Simulate 6 point distribution using crude method
    Z = six_point_crude_method(U)
    utils.histogram(Z, "6 point distribution", 5)

if __name__ == "__main__":
    main()
