#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-29 14:05:30
# @Author  : Roujia Li (liroujia0314@outlook.com)

import numpy as np
import matplotlib.pyplot as plt

def middle(data):
	mean_m=np.mean(data,0)
	m,n=np.shape(data)
	mean_data=np.tile(mean_m,(m,1))
	scaled_data=data-mean_data
	return scaled_data

def pca(data,k):
	mean_d=np.mean(data,0)
	mediated_data=middle(data)
	plt.scatter(mediated_data[:,0],mediated_data[:,1])
	covD=np.cov(mediated_data.T)
	feat_value, feat_vec=  np.linalg.eig(covD)#vec is in row
	plt.plot([feat_vec[:,0][0],0],[feat_vec[:,0][1],0],color='red')
	plt.plot([feat_vec[:,1][0],0],[feat_vec[:,1][1],0],color='red')
	plt.title('mediated data & the direction of principal component')
	plt.show()
	index = np.argsort(-feat_value)
	m,n=np.shape(data)
	if k > n:
		print("k is supposed to be lower than feature number")
		return
	else:
		comp_vec=np.matrix(feat_vec.T[index[:k]])#特征向量
		#print(comp_vec)
		final_data = mediated_data * comp_vec.T#主成分
		restore_data=(final_data * comp_vec) + mean_d#投影回去
		return final_data,restore_data

def figureresult(data1,data2):
	if data1.shape[0]==1:
		x1=np.array(data1).T
		y1=np.zeros((1,x1.shape[1]))
	else:
		x1,y1=np.array(data1).T
		pass
	#x2,y2=np.array(data2).T
	plt.scatter(x1,y1,)
	#plt.scatter(x2,y2,c='r')
	plt.axis()
	plt.title("data scatter")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()
