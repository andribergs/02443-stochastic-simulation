from math import log, floor, factorial, sqrt
import statistics as st
import random as random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import stats_utils as utils 
    
def metropolis_hastings():
    n = 10000
    A = 8
    X = [0 for _ in range(n)]
    
    for t in range(n-1):
        x_prime = np.random.randint(0, 10 + 1)
        x_t = X[t]
        
        #f_x a proportional distribution to the desired distribution
        f_x_prime = (A**x_prime)/factorial(x_prime)
        f_x_t = (A**x_t)/factorial(x_t)
        
        acceptance_probability = min(1, f_x_prime/f_x_t)
        u = random.uniform(0,1)
        if u <= acceptance_probability:
            X[t+1] = x_prime
        else:
            X[t+1] = x_t
    return X

def metropolis_hastings_two_call_types():
    n=10000
    A_1 = 4
    A_2 = 4
    X = [0 for _ in range(n)]
    Y = [0 for _ in range(n)]
    
    for t in range(n-1):
        x_t, y_t = X[t], Y[t]
        x_prime, x_double_prime = np.random.randint(0, 10 + 1), np.random.randint(0, 10 + 1)
        
        #f_x a proportional distribution to the desired distribution
        f_x_prime = ((A_1**x_prime)/factorial(x_prime)) * ((A_2**x_double_prime)/factorial(x_double_prime))
        f_x_t = ((A_1**x_t)/factorial(x_t)) * ((A_2**y_t)/factorial(y_t))
        
        acceptance_probability = min(1, f_x_prime/f_x_t)
        u = random.uniform(0,1)
        if u <= acceptance_probability:
            X[t+1] = x_prime
            Y[t+1] = x_double_prime
        else:
            X[t+1] = x_t
            Y[t+1] = y_t
    return (X, Y)

def metropolis_hastings_two_call_types_coordinate_wise():
    n=10000
    X = [0 for _ in range(n)]
    Y = [0 for _ in range(n)]
    Y[0] = np.random.randint(0, 10 + 1)
    m = actual_probability_matrix()
    
    for i in range(n):        
        x_n = m[Y[i]] / sum(m[Y[i]])
        x_t = np.random.choice(list(range(11)), size=1, replace=True, p=x_n)[0]
        
        y_n = m[:,x_t] / sum(m[:,x_t])
        y_t = np.random.choice(list(range(11)), size=1, replace=True, p=y_n)[0]
        
        X[i] = x_t
        Y[i] = y_t
    return (X, Y)

def actual_densities():
    n = 10
    A = 8
    d = []
    t = []
    for j in range(n):
        d.append(A**j / factorial(j))
    for i in range(n):
        t.append((A**i/factorial(i))/sum(d))
        
    return t

def actual_probability_matrix():
    n = 11
    A_1 = 4
    A_2 = 4
    m = np.zeros((n,n))
    K = calc_constant_K()
    for i in range(n):
        for j in range(n):
            if i+j <= n:
                m[i][j] = (1/K) * (A_1**j / factorial(j)) * (A_2**i)/factorial(i)
            else:
                m[i][j] = 0
    return m
        
def calc_constant_K():
    n = 11
    K = 0
    A_1 = 4
    A_2 = 4
    for i in range(n):
        for j in range(n):
            if i+j <= n:
                K = K + ((A_1**j / factorial(j)) * (A_2**i)/factorial(i))
    return K

def main():    
    
    #Expected densities
    Y = actual_densities()
    
    #Generate values from a truncated Poisson distribution
    X = metropolis_hastings()

    #Generate values from a distribution representing two different call types directly
    X_joint_directly, Y_joint_directly = metropolis_hastings_two_call_types()
    
    #Generate values from a distribution representing two different call types coordinate wise
    X_joint_coordinate_wise, Y_joint_coordinate_wise = metropolis_hastings_two_call_types_coordinate_wise()
    
    #Histogram for X
    density_X, _, _ = plt.hist(X, 10, density=True, alpha=0.5, color="blue")
    plt.title("Generated values for X")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.show()
    
    #Chi squared test for X
    test_stat, p_chi = utils.chi_squared_test(density_X, Y)
    print("Chi squared test for X")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    #Histogram for X_joint_directly and Y_joint_directly
    plt.hist2d(X_joint_directly, Y_joint_directly, 9)
    plt.title("Generated values for X_joint_directly and Y_joint_directly")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()
    
    #Chi squared test for X_joint_directly and Y_joint_directly
    density_X_joint_directly = [x/10000 for x in utils.sort_values_to_bins(X_joint_directly)]
    test_stat, p_chi = utils.chi_squared_test(density_X_joint_directly, Y)
    print("Chi squared test for X_joint_directly")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    density_Y_joint_directly = [y/10000 for y in utils.sort_values_to_bins(Y_joint_directly)]
    test_stat, p_chi = utils.chi_squared_test(density_Y_joint_directly, Y)
    print("Chi squared test for Y_joint_directly")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    #Histogram for X_joint_coordinate_wise and Y_joint_coordinate_wise
    plt.hist2d(X_joint_coordinate_wise, Y_joint_coordinate_wise, 9)
    plt.title("Generated values for X_joint_coordinate_wise and Y_joint_coordinate_wise")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()
    
    #Chi squared test for X_joint_coordinate_wise and Y_joint_coordinate_wise
    density_X_joint_coordinate_wise = [x/10000 for x in utils.sort_values_to_bins(X_joint_coordinate_wise)]
    test_stat, p_chi = utils.chi_squared_test(density_X_joint_coordinate_wise, Y)
    print("Chi squared test for X_joint_coordinate_wise")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))
    
    density_Y_joint_coordinate_wise = [y/10000 for y in utils.sort_values_to_bins(Y_joint_coordinate_wise)]
    test_stat, p_chi = utils.chi_squared_test(density_Y_joint_coordinate_wise, Y)
    print("Chi squared test for Y_joint_coordinate_wise")
    print("Test stat: {0}, p_value: {1}".format(test_stat, p_chi))

if __name__ == "__main__":
    main()
