#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-01 16:55:01
# @Author  : Roujia Li (liroujia0314@outlook.com)

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def middle(data):
	mean_m=np.mean(data,0)
	m,n=np.shape(data)
	mean_data=np.tile(mean_m,(m,1))
	scaled_data=data-mean_data
	return scaled_data

def facepca(data,k):
	mean_d=np.mean(data,0)
	mediated_data=np.mat(middle(data))
	covD=mediated_data*mediated_data.T
	feat_value, feat_vec= np.linalg.eig(covD)#vec is in row
	index = np.argsort(-feat_value)
	m,n=np.shape(data)
	if k > n:
		print("k is supposed to be lower than feature number")
		return
	else:
		comp_vec=np.matrix(mediated_data.T*feat_vec.T[index[:k]].T)#特征向量(32256,40)*(40,20)
		for i in range(k):
			comp_vec[:,i]=comp_vec[:,i]/np.linalg.norm(comp_vec[:,i])#特征向量归一化
		print('compvecdemension',comp_vec.shape)
		final_data = mediated_data * comp_vec#主成分(40,32256)*(32256,20)
		print('compdatademension',final_data.shape)
		#restore_data=(final_data * comp_vec.T) + mean_d#投影回去(40,20)*(20,32256)
		restore_data=(final_data * comp_vec.T) + np.tile(mean_d,(m,1))
		return mean_d,mediated_data,final_data,restore_data,comp_vec


def lodaimage(filepath):
    FaceMat = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.pgm':
                #print(os.path.join(root, file))
                im = Image.open(os.path.join(root, file))
                im = im.convert('L')
                FaceMat.append(np.array(im).flatten())
    return np.array(FaceMat)


if __name__ == '__main__':
    filepath = "./yaleB06/"
    facedata=lodaimage(filepath)
    print(facedata.shape)

di=2
mean_data,middle_data,comp_data,restore_data,eigen_vec=facepca(facedata,di)

restore=[]
face=[]
middle=[]
eigen=[]
mean_data=mean_data.reshape(192,168)
for j in range(0,di):
	eigen.append(eigen_vec.T[j,:].reshape(192,168))
	pass
for i in range(0,40):
	#comp_data[i,:].reshape(,6)
	restore.append(restore_data[i,:].reshape(192,168))
	face.append(facedata[i,:].reshape(192,168))
	middle.append(middle_data[i,:].reshape(192,168))
	pass

#formal faces
plt.figure(1)
for i in range(1,41):
    plt.subplot(4,10,i)
    plt.imshow(Image.fromarray(face[i-1]),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
#plt.show()

#pricipal components
plt.figure(2)
for i in range(1,41):
    plt.subplot(4,10,i)
    plt.imshow(Image.fromarray(restore[i-1]),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
#plt.show()

#restored faces
plt.figure(3)
for i in range(1,41):
    plt.subplot(4,10,i)
    plt.imshow(Image.fromarray(comp_data[i-1]),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])


plt.figure(4)
for i in range(1,41):
    plt.subplot(4,10,i)
    plt.imshow(Image.fromarray(middle[i-1]),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
#plt.show()
# plt.figure(5)
# plt.imshow(Image.fromarray(restore[0]),cmap=plt.cm.gray)

plt.figure(5)
for i in range(1,3):
    plt.subplot(1,2,i)
    plt.imshow(Image.fromarray(eigen[i-1]),cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
plt.show()
#Image.fromarray(mean_data).show()
