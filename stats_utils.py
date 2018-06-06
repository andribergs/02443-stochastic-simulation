from scipy import stats
import matplotlib.pyplot as plt

def kolmogorov_smirnov(rn, dist):
    return stats.kstest(rn, dist)

def chi_squared_test(observed, expected):
    return stats.chisquare(observed, f_exp=expected)

def histogram(x, title="Histogram", n_bins=10):
    N, bins, patches = plt.hist(x, n_bins)
    plt.title(title)
    plt.xlabel("Number of bins: {0}".format(n_bins))
    plt.ylabel("Amount in each bin")
    plt.show()
    return N, bins