import random as rnd
from math import sqrt
import numpy as np
import scipy as sp
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
    plt.figure()
    plt.title("Histogram")
    plt.show()

def scatterplot(rn):
    N = 100
    colors = np.random.rand(N)
    plt.figure()
    plt.scatter(rn[1:][0::N], rn[:-1][0::N], c=colors, alpha=0.5)
    plt.title("Scatterplot")
    plt.show()
    
def chi_squared_test(rn, n_classes=10):
    n_expected = len(rn) / n_classes
    step = 1 / n_classes
    total_stats = []
    for i in range(n_classes):
        n_observed = len([x for x in rn if x >= step*i and x < (step*i) + step])
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
    empirical = sorted(rn)
    hypothized = np.linspace(0, 1, n)
    D = max(abs(np.array(empirical) - np.array(hypothized)))
    test_stat =  (sqrt(n) + 0.12 + (0.11/sqrt(n))) * D
    p_value = 1.36 / sqrt(n)
    
    print("-----Kolmogorov–Smirnov test-----")
    print("Test stat: {0}".format(test_stat))
    print("Kolmogorov-smirnov dist p-value, with alpha = 0.05: {0}".format(p_value))
    check_result(test_stat, p_value)

def run_test_I(rn):
    mean_cutoff = st.mean(rn)
    run_size, all_runs = runs(rn)
    n1 = len([x for x in rn if x > mean_cutoff])
    n2 = len([x for x in rn if x < mean_cutoff])
    mean = 2*((n1*n2)/(n1+n2))+1
    variance = 2*((n1*n2*(2*n1*n2-n1-n2)) / ((n1+n2)**2 * (n1+n2-1)))
    test_stat = (len(run_size) - mean) / sqrt(variance)
    p_value = 1.64
    
    print("-----Run test I----")
    print("Test stat: {0}".format(test_stat))
    print("Normal distribution p-value, with alpha = 0.05 and Z-value = {0}: {1}".format(test_stat, p_value))
    check_result(test_stat, p_value)

def run_test_II(rn):
    n = len(rn)
    run_size, all_runs = runs(rn)
    runs_less_or_equal_six = [x for x in run_size if x <= 6]
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
    

def test_lcg():
    
    print("Testing LCG")
    
    #Values are pre-chosen to give some effect of randomness
    a = 129
    c = 26461
    M = 65536
    x0 = 1
    
    #Generate 10000 pseudo-random numbers
    rn = LCG(a,c,M,x0)
    
    run_tests(rn)

def test_system_available_generator():
    
    print("Testing random uniform built-in function")
    
    #Generate 10000 pseudo-random numbers
    rn = list(np.random.random_sample(10000))
    
    run_tests(rn)

    
# Utility functions
# -----------------
def run_tests(rn):
    #histogram
    histogram(rn)
    
    #scatterplot
    scatterplot(rn)
    
    print("\n")
        
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

def check_result(observed_stat, p_value):
    if observed_stat < p_value:
        print("Result: Test stat less than p_value -----> Passed ")
    else:
        print("Test stat greater than p_value -----> Failed ")
    print("\n")

def runs(rn):
    run_length = 1
    run_size = []
    run = []
    runs = []
    for i in range(len(rn)-1):
        run.append(rn[i])
        if rn[i] < rn[i+1]:
            run_length = run_length + 1
            if i == len(rn) - 2:
                run.append(rn[i+1])
        else:
            run_size.append(run_length)
            run_length = 1
            runs.append(run)
            run = []
            
    run_size.append(run_length)
    runs.append(run)
    
    return (run_size, runs)
# -----------------
    
def main():
    test_lcg()
    test_system_available_generator()

if __name__ == "__main__":
    main()
