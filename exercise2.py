from math import log, floor
import random as random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter


def geometric_distribution(U, p):
    return [w + 1 for w in [floor(z) for z in [ y/log(1-p) for y in [log(x) for x in U]]]]


def histogram_comparison_2d(x, y, title="Histogram", n_bins=10):
    plt.hist([x,y], n_bins, alpha=0.5, label=["Simulated","Expected"], color=["blue", "red"])
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.show()
    
def histogram_comparison_4d(x, y, z, w, title="Histogram", n_bins=10):
    plt.hist([x,y,z,w], n_bins, alpha=0.5, label=["Crude","Rejection", "Alias", "Expected"], 
             color=["blue", "red", "green", "purple"])
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.show()


def six_point_crude_method(U):
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    p_i_cumsum = np.cumsum(p_i).tolist()
    n = len(U)
    X = []
    for i in range(n):
        for j in range(len(p_i)):
            if U[i] <= p_i_cumsum[j]:
                X.append(j+1)
                break;
    return X
    
def six_point_rejection_method():
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    q_i = [1/6 for _ in range(6)]
    C = 2
    n = len(p_i)
    X = []
    while len(X) < 10000:
        y = floor(n * random.uniform(0,1)) + 1
        u2 = random.uniform(0,1)
        if u2 < (p_i[y-1]/C*q_i[y-1]):
            X.append(y)
    return X
        
    
def six_point_alias_method():
    #Generate F and L
    p_i = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
    n = len(p_i)
    F = [n*x for x in p_i]
    L = [x for x in range(1, len(p_i) + 1)]
    G = [i for i in range(len(F)) if F[i] >= 1]
    S = [i for i in range(len(F)) if F[i] <= 1]
    while len(S) != 0:
        k = G[0]
        j = S[0]
        L[j] = k
        F[k] = F[k] - (1 - F[j])
        if F[k] < 1:
            G = G[1:]
            S.append(k)
        S = S[1:]
    
    X = []
    while len(X) < 10000:
        y = floor(n * random.uniform(0,1)) + 1
        u2 = random.uniform(0,1)
        if u2 < F[y-1]:
            X.append(y)
        else:
            X.append(L[y-1])
    return X

def sort_values_to_bins(X, n_bins):
    c_X = Counter(X)
     # for making scipy.stats.chisquare test work properly
    n_observed = [c_X[x] if c_X[x] > 5 else 5 for x in range(1, n_bins + 1)]
    return n_observed
    

def main():
    #Generate 10000 pseudo-random numbers
    U = list(np.random.random_sample(10000))
    
    #Choose a value for p in the geometric distribution and simulate 10,000 outcomes.
    p = 0.5
    X = geometric_distribution(U, p)
    Y = stats.geom.rvs(p, size=10000)
    histogram_comparison_2d(X, Y, "Geometric Distribution", max(max(X), max(Y)) - 1)

    n_bins = max(max(X),max(Y))
    test_stat, p_chi = stats.chisquare(sort_values_to_bins(X, n_bins), 
                                       sort_values_to_bins(Y, n_bins))
    print("Chi squared test")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    
    #Simulate 6 point distribution using the crude, rejection and alias methods
    Z_crude = six_point_crude_method(U)
    Z_rejection = six_point_rejection_method()
    Z_alias = six_point_alias_method()
    Z_expected = stats.rv_discrete(name='6point', 
                                   values=(np.arange(1,7), (7/48, 5/48, 1/8, 1/16, 1/4, 5/16))).rvs(size=10000)
    histogram_comparison_4d(Z_crude, Z_rejection, Z_alias, Z_expected, "6 point distribution", 5)
    
    n_bins_crude = max(max(Z_crude), max(Z_expected))
    test_stat, p_chi = stats.chisquare(sort_values_to_bins(Z_crude, n_bins_crude), 
                                       sort_values_to_bins(Z_expected, n_bins_crude))
    print("Chi squared test for Crude")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    n_bins_rejection = max(max(Z_rejection), max(Z_expected))
    test_stat, p_chi = stats.chisquare(sort_values_to_bins(Z_rejection, n_bins_rejection), 
                                       sort_values_to_bins(Z_expected, n_bins_rejection))
    print("Chi squared test for Rejection")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    n_bins_alias = max(max(Z_expected), max(Z_expected))
    test_stat, p_chi = stats.chisquare(sort_values_to_bins(Z_alias, n_bins_alias), 
                                       sort_values_to_bins(Z_expected, n_bins_alias))
    print("Chi squared test for Alias")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))

if __name__ == "__main__":
    main()
