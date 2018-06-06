#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:23:38 2018

@author: thorsteinngj
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

def seedLCG(inVal):
    global rand
    rand = inVal

#Want M, c as relative prime.
#Want for each prime p of M, mod(a,p) = 1
#want

#The linear congruential generator
def lcg():
    a = 1664525 #129 good
    M = 2**32 #65536 good
    b = M
    c = 1013904223 #26461 good
    global rand
    rand = (a*rand+c) % b
    return rand / b

seedLCG(1)

x = []
for i in range(10000):
    x.append(lcg())
    
def histogram(x, n_bins=10):
    colors = ['tomato','papayawhip','plum','palegoldenrod','moccasin','khaki',
              'firebrick','tan','darksalmon','lime']
    N, bins, patches = plt.hist(x,n_bins)
    for i in range(len(patches)):
        patches[i].set_facecolor(colors[i])
    plt.title('Histogram')
    plt.show()
plt.figure(1)
plt.hist(x,bins=10)

def khi_squared(x, n_class=10):
    #dof = n_class-1-m (m=0) so here 9
    p_value = 3.32
    n_exp = np.size(x)/n_class
    step = 1 / n_class
    total_stats = []
    for i in range(n_class):
        n_obs = np.size([e for e in x if e >= step*i and e < (step*i) + step])
        #print(n_obs)
        stat = ((n_obs - n_exp)**2) / n_exp
        total_stats.append(stat)
    observed_stat = sum(total_stats)
    print("-----Chi squared test-----")
    print("Observed stat: {0}".format(observed_stat))
    print("Chi squared dist p-value, with alpha = 0.05: {0}".format(p_value))
    if observed_stat > p_value:
        print("Does not pass chi-squared test")
    else:
        print("Does pass chi-squared test")  

"""def kolmogorov_smirnov(x):
    x_sort = sorted(x)
    y = 0
    diff = []
    for i in range(len(x_sort)):
        xy = 1*i
        diff = [diff, abs(xy-y)]
        y = y + 1/len(x)
        diff = [diff, abs(xy-y)]
    Dn = max(diff)
    pvalue = (np.sqrt(len(x))+0.12+(0.11/np.sqrt(len(x))))*Dn
    print("-----Kolmogorov Smirnov test-----")
    print("Kolmogorov-Smirnov p-value:".format(pvalue))
"""

def kolmogorov_smirnov_test(x):
    n = len(x)
    x_sort = sorted(x)
    S = np.linspace(0,1,n)
    D_vec = abs(S-x_sort)
    D = max(D_vec)
    #For alpha = 0.05, and n>>50
    p_value = 1.36/np.sqrt(n)
    test_stat = D
    print("-----Kolmogorov–Smirnov test-----")
    print("Test stat: {0}".format(test_stat))
    print('Kolmogorov-smirnov dist p-value, with alpha = 0.05 is {0}'.format(p_value)) 
    if D > p_value:
        print("Does not pass Kolmogorov-Smirnov test")
    else:
        print("Does pass Kolmogorov-Smirnov test")  
    
def run_above_below(x):
    N = len(x)
    n1 = sum((x>np.mean(x))==True)
    n2 = sum((x<np.mean(x))==True)
    b = 0
    w = np.mean(x)
    flagger = 'none'
    for i in x:
        if i < w and flagger != 'lower':
            b += 1
            flagger = 'lower'
        elif i > w and flagger != 'upper':
            b += 1
            flagger = 'upper'
        #print(b)
    medaltal = (2*n1*n2/N)+(1/2) #Is 1 in Bo's slides, but 1/2 somewhere online
    #http://www.oswego.edu/~lwahl/classes/csc454/site/runsWithMean.html
    #https://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node44.html
    dreifni = (2*n1*n2*(2*n1*n2-N))/((N**2)*(N-1))
    stadalfravik = np.sqrt(dreifni)
    Z0 = (b-medaltal)/stadalfravik
    #According to:
    #http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_HypothesisTest-Means-Proportions/BS704_HypothesisTest-Means-Proportions3.html
    Zu = 1.645
    Zl = -1.645
    print("-----Run Test 1-----") 
    print("Test stat: {0}".format(Z0))
    print("Run Test 1 Z-value with alpha = 0.05 is +-{0}".format(Zu))    
    if Zl < Z0 < Zu:
        print('Does pass Run Test 1')
    else:
        print('Doesn''t pass it')
    #Failure to reject the hypothesis of independence occurs when 
    #$-z_{\alpha/2} \le Z_0 \le z_{\alpha/2}$, where $\alpha$ is 
    #the level of significance.

def run_up_down(x):
    n = len(x)
    runs = [1]
    k = 0
    for i in range(n-1):
        if x[i] < x[i+1]:
            runs[k] = runs[k]+1
        else:
            k += 1
            runs.append(1)
    runs = Counter(runs).most_common(len(set(runs)))
    runs = sorted(runs)
    _, R = list(zip(*runs))
    R = np.matrix(np.array(R)).T
    #Þarf að geta gert ef það eru fleiri eða færri en sex gildi fyrir R-in.!!!
    if np.size(R) < 6:
        R = 1
    R = np.matrix(np.array(R)).T
    B = np.matrix(np.array([ 1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])).T
    A = np.array(
        [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
         [9044.9, 18097, 27139, 36187, 45234, 55789],
         [13568, 27139, 40721, 54281, 67852, 83685],
         [18091, 36187, 54281, 72414, 90470, 111580],
         [22615, 45234, 67852, 90470, 113262, 139476],
         [27892, 55789, 83685, 111580, 139476, 172860]])
    summa = R - n*B
    Z = (1/(n-6))*summa.T*A*summa
    Z = np.float(Z)
    #According to some table online:
    test_stat = 12.59
    print("-----Run Test 2-----")
    print("Test stat: {0}".format(test_stat))
    print("Run Test 2 Z-value with alpha = 0.05 is {0}".format(Z))
    if Z < test_stat:
        print('Does pass Run Test 2')
    else:
        print('Doesn''t pass it')


#Histogram
histogram(x)

#Scatterplot
x1 = x[0:-1]
x2 = x[1:]
plt.figure(2)
colors = ['tomato','papayawhip','plum','palegoldenrod','moccasin','khaki',
              'firebrick','tan','darksalmon','lime']
plt.scatter(x1,x2,s=15,alpha=1,c=colors,marker='P',)
plt.xlabel(r'$U_i$ $indices$')
plt.ylabel(r'U$_{i+1}$ indices')
plt.title(r'Plot of U$_i$ versus U$_{i+1}$')

#Ki-kvaðrat prof
khi_squared(x)

#Kolmogorov-smirnov:
kolmogorov_smirnov_test(x)

#Run Test 1:
run_above_below(x)

#Run Test 2:
#run_up_down(x)



#Generate LCG myself wow. Experiment with different a b c.
#x1 = mod(ax0+c,M)

#Evaluate quality of generators with histograms, scatter plots (x2, kolmogorov 
#smirnof run, correlation tests)

#System available generator and perform various statistical tests