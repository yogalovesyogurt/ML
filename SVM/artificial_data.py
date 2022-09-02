#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-26 19:39:40
# @Author  : Roujia Li (liroujia0314@outlook.com)

import numpy as np
import matplotlib.pyplot as plt

#the amount of datas in each group is N
def gauss_data(N):
    mean1 = [1.5,1.5]
    cov1 = [[1,0],[0,1]]
    data = np.round(np.random.multivariate_normal(mean1,cov1,N),3)
    data=np.append(data,np.ones(N).reshape(N,1),1)
    
    mean2 = [-1.5,-1.5]
    cov2 = [[1,0],[0,1]]
    data1=np.round(np.random.multivariate_normal(mean2,cov2,N),3)
    data1=np.append(data1,-np.ones(N).reshape(N,1),1)
    data = np.append(data,data1,0)

    return data

def save_data(data,filename):
    with open(filename,'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i,0])+','+str(data[i,1])+','+str(data[i,2])+'\n')

def load_data(filename):
    data = []
    with open(filename,'r') as file:
        for line in file.readlines():
            data.append([ float(i) for i in line.split(',')])
    return np.array(data)


# data = gauss_data(200)
# save_data(data,'data_gauss.txt')
result=load_data("tested_artificial.txt")
group_one=[]
group_two=[]
for i in range(result.shape[0]):
    if result[i,2]==1:
        group_one.append(result[i,[0,1]])
    else:
        group_two.append(result[i,[0,1]])

group_one=np.array(group_one)
group_two=np.array(group_two)
x1,y1 = group_one.T
x2,y2 = group_two.T
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='r')
plt.axis()
plt.title("data scatter")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

 