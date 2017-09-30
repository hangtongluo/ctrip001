# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:47:14 2017

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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
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




########################################################################################



test_data = test_data.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间'], axis=1)
test_data = test_data.drop(['飞行季度', '飞行月份'], axis=1)
test = xgb.DMatrix(test_data)

#del test_data
#gc.collect()


model = xgb.Booster({'nthread':-1}, model_file = 'xgbclassifier.model')
prob = model.predict(test)

#flag = sorted(prob,reverse=True)[19488]
submit = pd.read_csv(r'train/submission_sample.csv', \
                        nrows=None)
submit['prob'] = prob
#submit = submit[submit['prob'] > flag]      
submit.to_csv(r'submit/xgb/xgb_result.csv', index=False)


print('pre_xgb_model finishing...')


#0.316208

























