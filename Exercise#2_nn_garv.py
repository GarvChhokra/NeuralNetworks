#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 01:20:44 2022

@author: garvchhokra
"""

# Exercise 2
import numpy as np
import neurolab as nl
min_val = -0.6
max_val = 0.6
num_points = 10
"""using numpy seed generating constant data for model testing"""
np.random.seed(1)
"""Generating uniform random values in x and y using min and max values and total of 10(num_points) and then 
reshaping it into rows and columbs so that we can put it into one array"""
x = np.random.uniform(min_val, max_val, num_points)
x = x.reshape(num_points, 1)
y = np.random.uniform(min_val, max_val, num_points)
y = y.reshape(num_points, 1)
input_garv = np.append(x, y, axis=1)
feature1 = input_garv[:,0]
feature2 = input_garv[:,1]
"""generating output by adding two features simaltaniously"""
output_garv = feature1+feature2
output_garv = output_garv.reshape(num_points,1)

"""calculating min and maximum values of each column so that we can pass it into our neural network"""
dim1_min, dim1_max = input_garv[:,0].min(), input_garv[:,0].max()
dim2_min, dim2_max = input_garv[:,1].min(), input_garv[:,1].max()

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

nn = nl.net.newff([dim1, dim2], [5, 3, 1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(input_garv, output_garv, epochs=1000, show=100, goal=0.00001)

print('\nTest results:')
data_test = [[0.1,0.2]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])
