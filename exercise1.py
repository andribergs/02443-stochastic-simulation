from random import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


def LCG(a, c, M, x0, amount_to_generate=10000):
    x = [x0]
    for i in range(amount_to_generate):
        x.append((a*x[i] + c) % M)
    return [e / M for e in x[1:]]

def histogram(x, n_bins=10):
    colors = ['blue', 'yellow', 'green', 'black', 'orange', 'purple', 'pink','red', 'tan', 'lime']
    N, bins, patches = plt.hist(x, n_bins)
    for i in range(len(patches)):
        patches[i].set_facecolor(colors[i])
    plt.title("Histogram")
    plt.show()

def scatterplot(rn):
    N = 100
    colors = np.random.rand(N)
#    plt.scatter([random() for i in range(, rn[0::N], c=colors, alpha=0.5)
    plt.scatter(rn[1:][0::N], rn[:-1][0::N], c=colors, alpha=0.5)
    plt.title("Scatterplot")
    plt.show()
    
def chi_squared_test(rn, n_classes=10):
    p_value = 3.32511 #alpha = 0.05, 0.950 p-value for chi squared distribution.
    n_expected = len(rn) / n_classes
    step = 1 / n_classes
    total_stats = []
    for i in range(n_classes):
        n_observed = len([x for x in rn if x >= step*i and x < (step*i) + step])
        print(n_observed)
        stat = ((n_observed - n_expected)**2) / n_expected
        total_stats.append(stat)
    observed_stat = sum(total_stats)
    print("-----Chi squared test-----")
    print("Observed stat: {0}".format(observed_stat))
    print("Chi squared dist p-value, with alpha = 0.05: {0}".format(p_value))
    check_result(observed_stat, p_value)

def kolmogorov_smirnov_test(rn):
    n = len(rn)
    K_plus = sqrt(n) * max( [((rn.index(x)/n) - x) for x in rn] )
    K_minus = sqrt(n) * max( [(x - ((rn.index(x)-1)/n)) for x in rn] )
    D = max(K_plus, K_minus)
    p_value = 1.36
    observed_stat = D
    print("-----Kolmogorov–Smirnov test-----")
    print("Observed stat: {0}".format(observed_stat))
    print("Kolmogorov-smirnov dist p-value, with alpha = 0.05: {0}".format(p_value))
    check_result(observed_stat, p_value)
    
def check_result(observed_stat, p_value):
    if observed_stat < p_value:
        print("Result: Observed stat less than p_value -----> Passed ")
    else:
        print("Observed stat greater than p_value -----> Falied ")

def main():
    #Values are pre-chosen to give some effect of randomness
    a = 129
    c = 26461
    M = 65536
    x0 = 1
    
    #Generate 10000 pseudo-random numbers
    rn = LCG(a,c,M,x0)
    
    #histogram
    histogram(rn)
    
    #scatterplot
    scatterplot(rn)
    
    #chi squared test
    chi_squared_test(rn)
    print("\n")
    
    #kolmogorov_smirnov
    kolmogorov_smirnov_test(rn)
    
    #run test I
    
    #run test II

    #correlation test    

if __name__ == "__main__":
    main()

