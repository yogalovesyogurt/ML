#!/usr/bin/env python
# coding: utf-8


# In[1]:
#code all by myself

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def cca(x,y):
    data=np.hstack((x,y))
    cov_mat=np.mat(np.cov(data.T))
    print(cov_mat)
    p=np.shape(x)[1]
    q=np.shape(y)[1]
    c11=cov_mat[0:p,0:p]
    c12=cov_mat[0:p,p:p+q]
    c21=cov_mat[p:p+q,0:p]
    c22=cov_mat[p:p+q,p:p+q]
    cvalue,cvec=np.linalg.eig(c11)
    tc11=cvec*np.diag(1/np.sqrt(cvalue))*cvec.T
    A=tc11*c12*np.linalg.inv(c22)*c21*tc11
    Avalue,Avec=np.linalg.eig(A)
    indexA = np.argsort(-Avalue)
    Avec=np.matrix(Avec.T[indexA[:p]].T)
    A_coef=np.zeros((p,p))
    for i in range(0,p):
        A_coef[:,i]=np.dot((Avec[:,i]).T,tc11)
    #A_coef=(Avec.T*tc11)
    A_v=np.sqrt(np.abs(np.sort(-Avalue)))
    
    c2value,c2vec=np.linalg.eig(c22)
    tc22=c2vec*np.diag(1/np.sqrt(c2value))*c2vec.T
    B=tc22*c21*np.linalg.inv(c11)*c12*tc22
    Bvalue,Bvec=np.linalg.eig(B)
    indexB=np.argsort(-Bvalue)
    Bvec=np.matrix(Bvec.T[indexB[:q]].T)
    B_coef=np.zeros((q,q))
    for i in range(0,q):
        B_coef[:,i]=np.dot((Bvec[:,i]).T,tc22)
    #B_coef=(Bvec.T*tc22).T
    
    B_v=np.sqrt(np.abs(np.sort(-Bvalue)))
    
    u1=np.dot(x,A_coef[:,0])
    v1=np.dot(y,B_coef[:,0])
    #print(np.shape(u1),np.shape(v1))
    u2=np.dot(x,A_coef[:,1])
    v2=np.dot(y,B_coef[:,1])
    plt.scatter(x=preprocessing.scale(u1),y=preprocessing.scale(v1))
    plt.show()
    plt.scatter(x=preprocessing.scale(u2),y=preprocessing.scale(v2))
    plt.show()
    return A_coef,A_v,B_coef,B_v


# In[2]:


def kernel_trans(x,s):
    zscore = preprocessing.StandardScaler()
    x = zscore.fit_transform(x)
    m,n=np.shape(x)
    k=np.mat(np.zeros((m,m)))
    for i in range(0,m):
        for j in range(0,m):
            k[i,j]=np.dot(x[i,:]-x[j,:],(x[i,:]-x[j,:]).T)
    k = np.exp(k/(-2*s**2))
    return np.mat(k)


# In[3]:


def kcca(x,y):
    m,n=np.shape(x)
    I=(1/m)*np.mat(np.ones(m))*np.mat(np.ones(m)).T
    J=np.diag(np.ones(m))-I
    J=np.mat(J)
    Kx=kernel_trans(x,0.1)
    Ky=kernel_trans(y,0.1)
    M=(1/m)*Kx.T*J*Ky
    L=(1/m)*Kx.T*J*Kx+0.1*Kx
    N=(1/m)*Ky.T*J*Ky+0.1*Ky
    A=np.linalg.inv(L)*M*np.linalg.inv(N)*M.T
    Avalue,Avec=np.linalg.eig(A)
    indexA = np.argsort(-Avalue)
    Avec=np.matrix((Avec.T[indexA[:m]]).T)
    A_v=np.sqrt(np.abs(np.sort(-Avalue)))
    
    B=np.linalg.inv(N)*M.T*np.linalg.inv(L)*M
    Bvalue,Bvec=np.linalg.eig(B)
    indexB = np.argsort(-Bvalue)
    Bvec=np.matrix((Bvec.T[indexB[:m]]).T)
    B_v=np.sqrt(np.abs(np.sort(-Bvalue)))

    u1=np.dot(Kx,Avec[:,1])
    v1=np.dot(Ky,Bvec[:,1])
    print(np.shape(u1),np.shape(v1))
    u2=np.dot(Kx,Avec[:,2])
    v2=np.dot(Ky,Bvec[:,2])
    plt.scatter(x=preprocessing.scale(u1),y=preprocessing.scale(v1))
    plt.show()
    plt.scatter(x=preprocessing.scale(u2),y=preprocessing.scale(v2))
    plt.show()
    return Avec,A_v,Bvec,B_v


# In[54]:


#simulation1
import pandas as pd
S1_trainx=np.random.rand(150,2)
S1_trainy=np.random.rand(150,2)
S1_noise=np.round(np.random.multivariate_normal([0,0],[[0.01,0],[0,0.01]],150),3)
S1_testx=S1_trainx+S1_noise
S1_testy=S1_trainy+S1_noise
S1_Acoef,S1_Av,S1_Bcoef,S1_Bv=cca(S1_trainx,S1_trainy)
S1_u1=np.dot(S1_testx,S1_Acoef[:,0])
S1_v1=np.dot(S1_testy,S1_Bcoef[:,0])
S1_u2=np.dot(S1_testx,S1_Acoef[:,1])
S1_v2=np.dot(S1_testy,S1_Bcoef[:,1])
plt.scatter(x=S1_u1,y=S1_v1)
plt.show()
plt.scatter(x=S1_u2,y=S1_v2)
plt.show()
print('训练集典型相关系数为')
print(S1_Av)
print('测试集典型相关系数为')
print(pd.Series(S1_u1).corr(pd.Series(S1_v1)),pd.Series(S1_u1).corr(pd.Series(S1_v2)))
print(pd.Series(S1_u2).corr(pd.Series(S1_v1)),pd.Series(S1_u2).corr(pd.Series(S1_v2)))


# In[55]:


plt.scatter(x=S1_trainx[:,0],y=S1_trainx[:,1])
plt.show()
plt.scatter(x=S1_trainy[:,0],y=S1_trainy[:,1])
plt.show()


# In[56]:


plt.scatter(x=S1_testx[:,0],y=S1_testx[:,1])
plt.show()
plt.scatter(x=S1_testy[:,0],y=S1_testy[:,1])
plt.show()


# In[57]:


S1_Akcoef,S1_Akv,S1_Bkcoef,S1_Bkv=kcca(S1_trainx,S1_trainy)
S1_tku1=np.dot(kernel_trans(S1_trainx,0.1),S1_Akcoef[:,1]).getA()
S1_tkv1=np.dot(kernel_trans(S1_trainy,0.1),S1_Bkcoef[:,1]).getA()
S1_tku2=np.dot(kernel_trans(S1_trainx,0.1),S1_Akcoef[:,2]).getA()
S1_tkv2=np.dot(kernel_trans(S1_trainy,0.1),S1_Bkcoef[:,2]).getA()
S1_ku1=np.dot(kernel_trans(S1_testx,0.1),S1_Akcoef[:,1]).getA()
S1_kv1=np.dot(kernel_trans(S1_testy,0.1),S1_Bkcoef[:,1]).getA()
S1_ku2=np.dot(kernel_trans(S1_testx,0.1),S1_Akcoef[:,2]).getA()
S1_kv2=np.dot(kernel_trans(S1_testy,0.1),S1_Bkcoef[:,2]).getA()
plt.scatter(x=S1_ku1,y=S1_kv1)
plt.show()
plt.scatter(x=S1_ku2,y=S1_kv2)
plt.show()
print('训练集典型相关系数为')
print(pd.Series(S1_tku1.reshape(150,)).corr(pd.Series(S1_tkv1.reshape(150,))),pd.Series(S1_tku2.reshape(150,)).corr(pd.Series(S1_tkv2.reshape(150,))))
print('测试集典型相关系数为')
print(pd.Series(S1_ku1.reshape(150,)).corr(pd.Series(S1_kv1.reshape(150,))),pd.Series(S1_ku1.reshape(150,)).corr(pd.Series(S1_kv2.reshape(150,))))
print(pd.Series(S1_ku2.reshape(150,)).corr(pd.Series(S1_kv1.reshape(150,))),pd.Series(S1_ku2.reshape(150,)).corr(pd.Series(S1_kv2.reshape(150,))))


# In[51]:


mean1 = [0,0]
cov1 = [[0.5,0],[0,0.5]]
data1 = np.round(np.random.multivariate_normal(mean1,cov1,150),3)
mean2 = [0,0,0]
cov2 = [[1,0,0],[0,1,0],[0,0,1]]
data2 = np.round(np.random.multivariate_normal(mean2,cov2,150),3)
#mean3 = [0,0]
#cov3 = [[0.5,0],[0,0.5]]
#data3 = np.round(np.random.multivariate_normal(mean3,cov3,150),3)
#Data=np.vstack((data1,data2))
#plt.scatter(x=Data[:,0],y=Data[:,1])
#plt.show()
#print(np.shape(Data))
cca(data1,data2)


# In[154]:


gaussAvec,gaussAv,gaussBvec,gaussBv=kcca(data1,data2)


# In[151]:


D = np.loadtxt("data.txt", usecols=np.arange(1,5)) 
D1=D[:,[0,1]]
D2=D[:,[2,3]]
cca(D1,D2)


# In[28]:





# In[ ]:





# In[ ]:





# In[ ]:




