#!/usr/bin/env python
# coding: utf-8

# In[22]:


import tensorflow as tf
from tensorflow import keras
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


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

(train_images, train_labels), (test_images, test_labels) = read_data()


# In[20]:


test=test_images

train_images = train_images.reshape([-1,28,28,1]) / 255.0
test_images = test_images.reshape([-1,28,28,1]) / 255.0


# In[24]:


class_names = ['T-shirt', 'pants', 'overpull', 'dress', 'coat',
              'sandal', 'shirt', 'sneakers','bag', 'boots']


# In[11]:


model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1),filters=32,kernel_size=5,strides=1,padding='same'),
    #(-1,28,28,32)->(-1,14,14,32)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,14,14,32)->(-1,14,14,64)
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'), 
    #(-1,14,14,64)->(-1,7,7,64)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,7,7,64)->(-1,7*7*64)
    keras.layers.Flatten(),
    #(-1,7*7*64)->(-1,256)
    keras.layers.Dense(256, activation=tf.nn.relu),
    #(-1,256)->(-1,10)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print(model.summary())


# In[12]:


lr = 0.001
model.compile(optimizer=tf.train.AdamOptimizer(lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[14]:


model.fit(train_images, train_labels, batch_size = 200, epochs=5,validation_data=[test_images[:1000],test_labels[:1000]])


# In[15]:


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('测试损失：%f 测试准确率: %f' % (test_loss, test_acc))


# In[26]:


predictions = model.predict(test_images)
plt.figure(figsize=(10,10))
 
# 预测 25 张图像是否准确，不准确为红色。准确为蓝色
for i in range(25,50):
    # 创建分布 5 * 5 个图形
    plt.subplot(5, 5, i-25+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 显示照片，以cm 为单位。
    plt.imshow(test[i], cmap=plt.cm.binary)
    
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




