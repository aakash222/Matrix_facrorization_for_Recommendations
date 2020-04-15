#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:54:40 2020

@author: smoke
"""
import sys
import numpy as np
from sklearn.model_selection import train_test_split


def matrix_factorization(R, P, Q, K, steps, lr, lambda1):
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            eij = R[i][2] - np.dot(P[int(R[i][0]),:],Q[:,int(R[i][1])])
            for k in range(K):
                P[int(R[i][0])][k] = P[int(R[i][0])][k] + lr * (2 * eij * Q[k][int(R[i][1])] - lambda1 * P[int(R[i][0])][k])
                Q[k][int(R[i][1])] = Q[k][int(R[i][1])] + lr * (2 * eij * P[int(R[i][0])][k] - lambda1 * Q[k][int(R[i][1])]) 
    return P, Q.T

    

f_in = open("/home/smoke/Documents/ML/project/Datasets/HIN_dataset/Yelp/user_business.dat","r");
f_in0 = f_in.readlines()[1:];
a = list();
for i in f_in0:
    a.append(list(map(float,i.split())));
f_in.close();
print(len(a))

a.sort(key = lambda x: x[0])
a = np.array(a)
row = a[len(a)-1][0];
a = a.T
col = max(a[1]);
###### If index of user and business start at 1, then doing 0 indexing ##########
if min(x[0]) == 1:
    a[0] = list(map(lambda x : int(x-1), a[0]))
if min(x[1]) == 1:
    a[1] = list(map(lambda x : int(x-1), a[1]))

a = a.T
steps = 20;
lr = 0.002
lambda1 = 0.02
dim = int(20);

    
print(row,col)
row = int(row)
col = int(col)
U = np.random.rand(row, dim);
V = np.random.rand(dim, col);



in_matrix = [[0.0]*col for i in range(row)];
X_train, X_test = train_test_split(a, test_size = .1);


nU, nV = matrix_factorization(X_train,U,V,dim,steps,lr, lambda1);

print("factorization done!!!");

e = 0;
for i in range(len(X_test)):
    temp = np.dot(nU[int(X_test[i][0])], nV[int(X_test[i][1])]);
    e += pow(temp - X_test[i][2],2);
e /= len(X_test);
print('error ',e**0.5);

