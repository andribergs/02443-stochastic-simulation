#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:40:24 2018

@author: thorsteinngj
"""

import numpy as np
import pandas as pd
import random

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
    n = 10 #n = 6 eða 20
    while(random1 == random2):
        random1 = np.int(np.floor(np.random.uniform(0,n,1)))
        random2 = np.int(np.floor(np.random.uniform(0,n,1)))
    
    t = Sample
    t[random1],t[random2]=t[random2],t[random1]
    return t

#%% - Travelling salesman from exercise
Initial = np.arange(6)
Route = np.arange(6)
Initial = random.sample(range(6),6)
#Sample = my_shuffle(Route)
distmatTS = np.array([[0, 5,3,1,4,12],
                      [2,0,22,11,13, 30],
                      [6,8,0,13,12,5],
                      [33,9,5,0,60,17],
                      [1,15,6,10,0,14],
                      [24,6,8,9,40,0]])


InTot, InStates = obj_calc(Initial) 


#obj_counter = obj_calc(Sample)      

Tk = 1
k = 1
GB = 99999

Obj = 0


Obj1 = InTot
BestR = InStates
BestScore = Obj1
AllScore = InTot

for j in range(10000):
    temp = swap(Route)
    Obj2, Obj2States = obj_calc(Route)
    AllScore = np.append(AllScore,Obj2)
    print(j)
    #Check if we better the score
    if (Obj2 < Obj1):
        BestScore = np.append(BestScore,Obj2)
        Obj1 = Obj2
        BestR = Obj2States    
    else:
        Obj2 = Obj1
   
print("The best score we can obtain is {0}".format(Obj1))

#%% - Other cost matrix from internet
        

filePath = '/home/thorsteinngj/02443-stochastic-simulation/Verkefni/fylki.txt'

Initial = np.arange(20)
Route = np.arange(20)
#Sample = my_shuffle(Route)
cost = pd.read_table(filePath, sep = "\s+", header=0, index_col = 0)
distmatTS = np.array(cost)



ObjectiveCost = []
for l in range(10):
    TestState = random.sample(range(20),20)
    inObj, inRoute = obj_calc(TestState)
    Tk = 1
    k = 1
    
    Obj1 = inObj
    BestR = inRoute
    BestScore = inObj
    AllScore = inObj
    
    for j in range(10000):
        temp = swap(Route)
        Obj2, Obj2States = obj_calc(Route)  
        AllScore = np.append(AllScore,Obj2)
        print(j)
    #Check if we better the score
        if (Obj2 < Obj1):
            Obj1 = Obj2
            BestR = Obj2States
            BestScore = np.append(BestScore,Obj2)
        else:
            Obj2 = Obj1
    
    ObCost = Obj2
    ObjectiveCost = np.append(ObjectiveCost,Obj2)


#obj_counter = obj_calcsa(Sample)      

Tk = 1
k = 1
GB = 99999

Obj = 0




   
print("The best score we can obtain is {0}".format(Obj1))


#%%


"""
    exp_val = np.exp(-(obj_counter1-obj_counter_start)/Tk)
    if (obj_counter_start > obj_calc(temp)):
        Sample = temp
        obj_counter_start = obj_counter1
    elif ((np.random.uniform(0,1,1)) < exp_val):
        Sample = temp
        obj_counter_start = obj_counter1
    if (obj_counter_start < GB):
        GB = obj_counter_start
        Best = Sample
        objls = np.append(objls, GB)
    
    Tk = 1/np.sqrt(1+k)
    objects = np.append(objls,obj_counter_start)
    k = k*1.005


print(" ")
print(k)
print("The best score obtained is: {0}".format(GB)))
print("The best route obtained is: ")
print(Bestroute)
print("objective_list")
print(objective_list)
  plot(obj1s, main = "The Solution Path", ylab = "Value", xlab = "Nr Objective", ylim = c(0, max(obj1s)))
"""


#Implement simulated annealing for the travelling salesman.
#Have input be positions in plane of the n stations.
#Let the cost of going i to j be the Euclidian distance between
#station i and j.
#As cooling scheme, use e.g. Tk = 1/√(1 + k).
#The route must end where it started.
#Initialise with a random permutation of stations.
#As proposal, permute two random stations on the route.
#Plot the resulting route in the plane.
#Debug with stations on a circle. Then modify your progamme to
#work with costs directly and apply it to the cost matrix from the
#course homepage.