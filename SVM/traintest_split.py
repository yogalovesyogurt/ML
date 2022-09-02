#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-26 21:30:16
# @Author  : Roujia Li (liroujia0314@outlook.com)

from sklearn.model_selection import train_test_split
 
c = []
filename = 'data_gauss.txt'
out_train = open('train_artificial.txt','w')
out_test = open('test_artificial.txt','w')
 
for line in open(filename):
    c.append(line)
  
c_train,c_test = train_test_split(c,test_size = 0.3)
for i in c_train:
    out_train.write(i)
for i in c_test:
    out_test.write(i)
