# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:30:43 2021

@author: PatCa
"""

#Make simulation 2 with p=0.3 and one with p=0.1

import numpy as np
from scipy.stats import binom

def pobin():
    n_sample = 100000
    error_rate_list = []
    for i in range(100):
        a1 = np.random.binomial(1,0.3,n_sample)
        a2 = np.random.binomial(1,0.3,n_sample)
        a3 = np.random.binomial(1,0.1,n_sample)
        
        a = np.transpose(np.array([a1,a2,a3]))
        b = np.sum(a,axis=1)
        c = b[b>1.5]   
        error_rate = len(c)/n_sample
        
        error_rate_list.append(error_rate)

    ave_error_rate = sum(error_rate_list)/len(error_rate_list)    
    
    return ave_error_rate

def q2bin():
    n_sample = 100000
    error_rate_list = []
    for i in range(100):     
        a1 = np.random.binomial(1,0.3,n_sample)
        a2 = np.random.binomial(1,0.3,n_sample)
        a3 = np.random.binomial(1,0.3,n_sample)
        
        a = np.transpose(np.array([a1,a2,a3]))
        b = np.sum(a,axis=1)
        c = b[b>1.5]
        error_rate = len(c)/n_sample
    
        error_rate_list.append(error_rate)

    ave_error_rate = sum(error_rate_list)/len(error_rate_list)
    
    return ave_error_rate


