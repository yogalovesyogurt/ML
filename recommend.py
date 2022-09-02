#!/usr/bin/env python
# coding: utf-8

# In[3]:


from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf,Reader
import os

file_path = os.path.expanduser('data.csv')
reader = Reader(line_format='user item rating', sep=',')
data = Dataset.load_from_file(file_path, reader=reader)
# k折交叉验证(k=3)
data.split(n_folds=3)
algo = SVD()
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
#输出结果
print_perf(perf)


# In[4]:


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import io

from surprise import KNNBaseline

def read_item_names():
 

    file_name = os.path.expanduser('data.csv')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# 首先，用算法计算相互间的相似度
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)

rid_to_name, name_to_rid = read_item_names()


raw_id = name_to_rid['2']
inner_id = algo.trainset.to_inner_iid(raw_id)

neighbors = algo.get_neighbors(inner_id, k=10)

# Convert inner ids of the neighbors into names.
neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in neighbors)
neighbors = (rid_to_name[rid]
                       for rid in neighbors)

print()
print('The 10 nearest neighbors of cloth2 are:')
for clothes in neighbors:
    print(clothes)


# In[6]:





# In[ ]:




