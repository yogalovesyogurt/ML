#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-26 21:08:25
# @Author  : Roujia Li (liroujia0314@outlook.com)

from svm import *
from svmutil import *
import numpy as np


train_label,train_pixel = svm_read_problem('train_artificial_svm.txt') 
predict_label,predict_pixel = svm_read_problem('test_artificial_svm.txt')
prob  = svm_problem(train_label, train_pixel)
param = svm_parameter('-c 32768.0 -g 0.00048828125 -t 2')
model = svm_train(prob, param)
print("result:")
p_label, p_acc, p_val = svm_predict(predict_label, predict_pixel, model);

print(p_acc)
print("real label:")
print(predict_label[0:20])
print("svm predict label:")
print(p_label[0:20])
# fp=open('label.txt','w')
# for i in range(np.array(p_label).shape[0]):
# 	fp.write(str(p_label[i])+'\n')
