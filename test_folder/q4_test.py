# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:45:34 2021

@author: PatCa
"""

#Original
data1 = [('id1', 'addr1', 'pw1'), ('idx', 'addr1', 'pwx'),
        ('idz','addrz','pwz'),('idy','addry','pw1')]

#Empty list
data = [()]

#all different tuples
data = [('id1', 'addr1', 'pw1'), ('idx', 'addr21', 'pwx'),
        ('idz','addrz','pwz'),('idy','addry','pw12')]

#All tuples point to the same person
data1 = [('id1', 'addr1', 'pw1'), ('idx', 'addr1', 'pwx'),
        ('idz','addrz','pwx'),('idy','addr1','pw1')]

#Different tuple length
data = [('id1', 'addr1', 'pw1', 'a'), ('idx', 'addr1', 'pwx', 'a'),
        ('idz','addrz','pwz','d'),('idy','addry','pw1', 's')]

#Varying tuple length
data = [('id1', 'addr1', 'pw1', 'a'), ('idx', 'addr1', 'pwx', 'a'),
        ('idz','addrz','pwz','d'),('idy','addry', 's')]

#Varying data_type
data = [('id1', 5, 'pw1', 'a'), ('idx', 'addr1', 'pwz', 'a'),
        ('idz','addrz','pwz', 'banana'),('idy','addry', 'fed',3.4)]


import sys  
sys.path.insert(0, 'C:/Users/PatCa/Documents/PythonScripts/DBS/src')

import q4

idx_list = q4.list_fcn(data1)
print(idx_list)
