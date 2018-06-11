#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:40:24 2018

@author: thorsteinngj
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def my_shuffle(array):
    random.shuffle(array)
    return array

def obj_calc(Sample):
    states = np.append(Sample,Sample[0])
    n = np.size(states)
    obj_counter = 0
    for i in range (1,n):
         obj_counter = distmatTS[states[i-1],states[i]]+obj_counter
    return obj_counter, states

def swap(Sample):
    random1 = 0
    random2 = 0
    n = 20 #n = 6 or 20
    while(random1 == random2):
        random1 = np.int(np.floor(np.random.uniform(0,n,1)))
        random2 = np.int(np.floor(np.random.uniform(0,n,1)))
    
    t = Sample
    t[random1],t[random2]=t[random2],t[random1]
    return t

#%% - Travelling salesman from exercise
InState = np.arange(6)
TestState = random.sample(range(6),6)
#Sample = my_shuffle(Route)
distmatTS = np.array([[0, 5,3,1,4,12],
                      [2,0,22,11,13, 30],
                      [6,8,0,13,12,5],
                      [33,9,5,0,60,17],
                      [1,15,6,10,0,14],
                      [24,6,8,9,40,0]])
#%% - Other cost matrix from internet
        

filePath = '/zhome/2e/9/124284/Term2/02443-stochastic-simulation/Verkefni/fylki.txt'

cost = pd.read_table(filePath, sep = "\s+", header=0, index_col = 0)
distmatTS = np.array(cost)
Wow = []
TestState = random.sample(range(20),20)  #initial_states
for i in range(5):
    InState = np.arange(20)
    
    InRes,InRoute = obj_calc(InState) #calculate_obj (res)
    TestRes, TestRoute = obj_calc(TestState)
    
#Sample = my_shuffle(Route)
    #CurrRoute = TestRoute #CurrRoute hja hring
    OldRes = TestRes
    OldRoute = TestRoute
    AllRes = [TestRes]
    EveryRes = [TestRes]
    BestRes = [TestRes]
    globalBest = TestRes
    loc = [0]
    T = 1
    k = 1
    for j in range(10000):
        NewState = swap(TestState)
        NewRes, NewRoute = obj_calc(NewState)
        AllRes = np.append(AllRes, OldRes)
        Exp_stat = np.exp(-(NewRes-OldRes)/T)
        Rand = np.random.uniform(0,1,1)
        if (OldRes > NewRes):
            OldRes = NewRes
            TestState = NewState
            EveryRes = np.append(EveryRes, NewRes)
        elif (Rand < Exp_stat):
            OldRes = NewRes
            TestState = NewState
            EveryRes = np.append(EveryRes, NewRes)
        if NewRes < globalBest:
            globalBest = OldRes
            BestRoute = TestState
            BestRes = np.append(BestRes, globalBest)
            EveryRes = np.append(EveryRes, NewRes)
            loc.append(j)
    
        T = T/np.sqrt(1+k)
        k = k*1.005
    Res = BestRoute        
    TestState = BestRoute
    Wow = np.append(Wow,BestRes[-1])
        
print("-----Travelling Salesman with 20 cities-----")
print("The best score we can obtain is {0}".format(BestRes[-1]))
Result = np.append(BestRoute,BestRoute[0])

#%%
x = np.arange(np.size(AllRes))
plt.figure()
plt.plot(x,AllRes,'k*')
plt.xlabel('Number of iterations')
plt.ylabel('Score')

#%%
def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transfopath1rm it into a list of ordered 
    # coordinates

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    
    plt.plot(x, y, 'co')


    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                    head_width = a_scale, color = 'r', 
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(min(x)*1.1, max(x)*1.1)
    plt.ylim(min(y)*1.1, max(y)*1.1)
    plt.show()

x_cor =[0,2,4,6,8,10,12,14,16,18,20,18,16,14,12,10,8,6,4,2]
y_cor = [0,2,4,6,8,10,12,14,16,18,0,-18,-16,-14,-12,-10,-8,-6,-4,-2]
points = []
for i in range(0, len(x_cor)):
    points.append(([x_cor[i],y_cor[i]]))
    
"path3 = list(InRoute[0:-1])
#path2 = list(TestRoute)
path1 = list(Result[0:-1])

paths = [path1]

plotTSP(paths,points,1)
