#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-27 08:39:58
# @Author  : Roujia Li (liroujia0314@outlook.com)

from svm import *
from svmutil import *
#未归一化
# train_label,train_pixel = svm_read_problem('train_blood_svm.txt') 
# predict_label,predict_pixel = svm_read_problem('test_blood_svm.txt')
# prob  = svm_problem(train_label, train_pixel)
# param = svm_parameter('-c 2048.0 -g 3.0517578125e-05 -h 0 -t 2')
# model = svm_train(prob, param)
# print("result:")
# p_label, p_acc, p_val = svm_predict(predict_label, predict_pixel, model);
# print(p_acc)
# print("real label:")
# print(predict_label[0:20])
# print("svm predict label:")
# print(p_label[0:20])

#归一化处理后
train_label,train_pixel = svm_read_problem('train_blood_svm_one.txt') 
predict_label,predict_pixel = svm_read_problem('test_blood_svm_one.txt')
prob  = svm_problem(train_label, train_pixel)
param = svm_parameter('-c 8192.0 -g 0.5 -t 2 -h 0')
model = svm_train(prob, param)
print("result:")
p_label, p_acc, p_val = svm_predict(predict_label, predict_pixel, model);
print(p_acc)
print("real label:")
print(predict_label[0:20])
print("svm predict label:")
print(p_label[0:20])