#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-29 15:02:00
# @Author  : Roujia Li (liroujia0314@outlook.com)

import numpy as np
from pca_function import *
import pandas as pd
import matplotlib.pyplot as plt

mean1 = [1,2]
cov1 = [[0.5,0],[0,0.5]]
data1 = np.round(np.random.multivariate_normal(mean1,cov1,150),3)
mean2 = [0,0]
cov2 = [[1,0],[0,1]]
data2 = np.round(np.random.multivariate_normal(mean2,cov2,150),3)
data=np.vstack((data1,data2))
x_oringin,y_oringin=data.T
plt.figure(1)
plt.scatter(x_oringin,y_oringin)
plt.axis()
plt.title("original data")
plt.xlabel("x")
plt.ylabel("y")
#plt.show()
x_mediate,y_mediate=middle(data).T
plt.figure(2)
plt.scatter(x_mediate,y_mediate)
plt.axis()
plt.title("mediated original data")
plt.xlabel("x")
plt.ylabel("y")
#plt.show()

plt.figure(3)
comp_data,restore_data=pca(data,1)

plt.figure(4)
if comp_data.shape[1]==1:
	x1=np.array(comp_data).T
	y1=np.zeros((1,comp_data.shape[0]))
else:
	x1,y1=np.array(comp_data).T
	pass
plt.scatter(x1,y1)
plt.scatter(x_oringin,y_oringin,c='r')
plt.axis()
plt.title("oringinal & projection data")
plt.xlabel("x")
plt.ylabel("y")
#plt.show()

x2,y2=np.array(restore_data).T
plt.figure(5)
plt.scatter(x_oringin,y_oringin,c='r')
plt.scatter(x2,y2,marker='v')
plt.axis()
plt.title("oringinal & restore data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#another data
def loaddata(datafile):
    return np.array(pd.read_csv(datafile,sep="\t",header=-1)).astype(np.float)
dataone=loaddata("data.txt")
comp_data2,restore_data2=pca(dataone,2)
figureresult(comp_data2,restore_data2)