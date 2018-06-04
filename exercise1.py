from random import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import statistics as st


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
    n_expected = len(rn) / n_classes
    step = 1 / n_classes
    total_stats = []
    for i in range(n_classes):
        n_observed = len([x for x in rn if x >= step*i and x < (step*i) + step])
        print(n_observed)
        stat = ((n_observed - n_expected)**2) / n_expected
        total_stats.append(stat)
    test_stat = sum(total_stats)
    p_value = 3.32511 #alpha = 0.05, 0.950 p-value for chi squared distribution n_classes - 1 df.
    
    print("-----Chi squared test-----")
    print("Test stat: {0}".format(test_stat))
    print("Chi squared dist p-value, with alpha = 0.05 and {0} df: {1}".format(n_classes - 1, p_value))
    check_result(test_stat, p_value)

def kolmogorov_smirnov_test(rn):
    n = len(rn)
    K_plus = sqrt(n) * max( [((rn.index(x)/n) - x) for x in rn] )
    K_minus = sqrt(n) * max( [(x - ((rn.index(x)-1)/n)) for x in rn] )
    D = max(K_plus, K_minus)
    test_stat = D
    p_value = 1.36
    
    print("-----Kolmogorov–Smirnov test-----")
    print("Test stat: {0}".format(test_stat))
    print("Kolmogorov-smirnov dist p-value, with alpha = 0.05: {0}".format(p_value))
    check_result(test_stat, p_value)

def run_test_I(rn):
    mean = st.mean(rn)
    n1 = len([x for x in rn if x > mean])
    n2 = len([x for x in rn if x < mean])
    runs = run_count(rn)
    mu = 2*((n1*n2)/(n1+n2))+1
    sigma2 = 2*((n1*n2*(2*n1*n2-n1-n2)) / ((n1+n2)**2 * (n1+n2-1)))
    test_stat = (runs - mu) / sqrt(sigma2)
    p_value = "???"
     
    print("-----Run test I----")
    print("Test stat: {0}".format(test_stat))
    print("Normal distribution p-value, with alpha = 0.05: {0}".format(p_value))

def run_test_II(rn):
    n = len(rn)
    runs_less_or_equal_six = [x for x in runs(rn) if x <= 6]
    R = np.array([runs_less_or_equal_six.count(r) for r in range(1,7)])
    B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    A = np.array(
            [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
             [9044.9, 18097, 27139, 36187, 45234, 55789], 
             [13568, 27139, 40721, 54281, 67852, 83685], 
             [18091, 36187, 54281, 72414, 90470, 111580], 
             [22615, 45234, 67852, 90470, 113262, 139476], 
             [27892, 55789, 83685, 111580, 139476, 172860]], dtype=float)
    vector_term = R - n*B
    test_stat = (1/(n-6)) * np.dot(vector_term.T, np.dot(A, vector_term))
    p_value = 1.64 #alpha = 0.05, 0.950 p-value for chi squared distribution with 6 df.
    
    print("-----Run test II----")
    print("Test stat: {0}".format(test_stat))
    print("Chi squared dist p-value, with alpha = 0.05 and 6 df: {0}".format(p_value))
    check_result(test_stat, p_value)

def correlation_test(rn):
    a = 1 + 1
    
    
# Utility functions
# -----------------
def check_result(observed_stat, p_value):
    if observed_stat < p_value:
        print("Result: Test stat less than p_value -----> Passed ")
    else:
        print("Test stat greater than p_value -----> Falied ")
    print("\n")

def runs(rn):
    run_length = 1
    runs = []
    for i in range(len(rn)-1):
        if rn[i] < rn[i+1]:
            run_length = run_length + 1
        else:
            runs.append(run_length)
            run_length = 1
    runs.append(run_length)
    return runs

def run_count(rn):
    return len(runs(rn))
# -----------------
    
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
    
    #kolmogorov_smirnov
    kolmogorov_smirnov_test(rn)
    
    #run test I
    run_test_I(rn)
    
    #run test II
    run_test_II(rn)
    
    #correlation test   
    correlation_test(rn)

if __name__ == "__main__":
    main()

