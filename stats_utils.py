from scipy import stats
import matplotlib.pyplot as plt

def kolmogorov_smirnov(rn, dist="norm"):
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

def sort_values_to_bins(v, n_bins=10):
    step = max(v) / n_bins
    n_observed = []
    for i in range(n_bins):
        n_observed.append(len([x for x in v if x >= step*i and x < (step*i) + step]))
    return n_observed