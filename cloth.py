#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import matplotlib


# In[1]:


def read_data():
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for i in range(len(files)):
         paths.append('./'+ files[i])
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
 
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
 
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
 
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
        
    return (x_train, y_train), (x_test, y_test)


# In[3]:


(train_images, train_labels), (test_images, test_labels) = read_data()


# In[4]:


class_names = ['T-shirt', 'pants', 'overpull', 'dress', 'coat',
              'sandal', 'shirt', 'sneakers','bag', 'boots']


# In[25]:



# 创建一个新图形
plt.figure()
 
# 显示一张图片在二维的数据上 train_images[0] 第一张图
plt.imshow(train_images[8])
 
# 在图中添加颜色条
plt.colorbar()
 
# 是否显示网格线条,True: 显示，False: 不显示
plt.grid(False)


# In[6]:



train_images = train_images / 255.0
 
test_images = test_images / 255.0


# In[13]:


# 保存画布的图形，宽度为 10 ， 长度为10
plt.figure(figsize=(10,10))
 
# 显示训练集的 25 张图像
for i in range(25,50):
    # 创建分布 5 * 5 个图形
    plt.subplot(5, 5, i-25+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 显示照片，以cm 为单位。
    plt.imshow(train_images[i], cmap=plt.cm.binary)
 
    # 此处就引用到上面的中文字体，显示指定中文，对应下方的图片意思，以证明是否正确
    plt.xlabel(class_names[train_labels[i]])


# In[38]:


# 建立模型
def build_model():
    # 线性叠加
    model = tf.keras.models.Sequential()
    # 改变平缓输入 
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 第一层紧密连接128神经元
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    # 第二层分10 个类别
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model


# In[39]:



model = build_model()
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[40]:


model.fit(train_images, train_labels, epochs=5)


# In[41]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print('测试损失：%f 测试准确率: %f' % (test_loss, test_acc))


# In[42]:


predictions = model.predict(test_images)
for i in range(25,50):
    pre = class_names[np.argmax(predictions[i])]
    tar = class_names[test_labels[i]]
    print("预测：%s   实际：%s" % (pre, tar))


# In[43]:



plt.figure(figsize=(10,10))
 
# 预测 25 张图像是否准确，不准确为红色。准确为蓝色
for i in range(25,50):
    # 创建分布 5 * 5 个图形
    plt.subplot(5, 5, i-25+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 显示照片，以cm 为单位。
    plt.imshow(test_images[i]*255, cmap=plt.cm.binary)
    
    # 预测的图片是否正确，黑色底表示预测正确，红色底表示预测失败
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'black'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
                                color=color)
plt.show()


# In[ ]:




