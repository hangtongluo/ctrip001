# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:46:38 2017

@author: Administrator
"""

import gc
from collections import Counter
from datetime import datetime 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
import xgboost as xgb
import lightgbm as lgb
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})


test_data = pd.read_csv(r'features/test_data.csv', nrows=None, encoding='gbk')
print(test_data.shape)
########################################################################################
#rate = pd.read_csv(r'features/rate1.csv', nrows=None, encoding='gbk')
#test_data = pd.merge(test_data, rate, on='航班编号', how='left')



########################################################################################



test_data = test_data.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间'], axis=1)
test_data = test_data.drop(['飞行季度', '飞行月份'], axis=1)
test = test_data.values


del test_data
gc.collect()


model = lgb.Booster(model_file='lgbclassifier.txt') #init model
prob = model.predict(test, num_iteration=model.best_iteration)



submit = pd.read_csv(r'train/submission_sample.csv', \
                        nrows=None)
submit['prob'] = prob
submit.to_csv(r'submit/lgb/result.csv', index=False)


print('pre_lgb_model finishing...')





















