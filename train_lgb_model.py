# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 00:20:18 2017

@author: Administrator
"""

import gc
from collections import Counter
from datetime import datetime 
import time
from dateutil.parser import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})


train_data = pd.read_csv(r'features/train_data.csv', nrows=None, encoding='gbk')
train_data['计划起飞时间'] = pd.to_datetime(train_data['计划起飞时间'])
print(train_data.shape)

train_target = train_data['航班是否取消']
train_data = train_data.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间','航班是否取消'], axis=1)
train_data = train_data.drop(['飞行季度', '飞行月份'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_target, test_size=0.2, random_state=2017)

del train_target, train_data
gc.collect()

lgb_train = lgb.Dataset(X_train.values, y_train.values)
lgb_val = lgb.Dataset(X_val.values, y_val.values)

del X_train, y_train, X_val, y_val
gc.collect()

params={'boosting_type':'gbdt',
	    'objective': 'binary',
	    'metric':'auc',
	    'max_depth':6,
	    'num_leaves':2**6, #80, 
	    'lambda_l2':1,
	    'subsample':0.7, #0.7
	    'learning_rate': 0.1,
	    'feature_fraction':0.7, #0.7 
	    'bagging_fraction':0.8,
	    'bagging_freq':10,
	    'num_threads':-1,
        'seed':2017,
#        'min_data_in_leaf': 100
}

#模型训练
model = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_val,early_stopping_rounds=20)
#model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=lgb_val,early_stopping_rounds=50)
model.save_model('lgbclassifier.txt')  #save model
  

train_pre = model.predict(X_train.values, num_iteration=model.best_iteration)
print('train_pr_auc：',average_precision_score(y_train.values, train_pre))
val_pre = model.predict(X_val.values, num_iteration=model.best_iteration)
print('val_pr_auc：',average_precision_score(y_val.values, val_pre))

del X_train, y_train, X_val, y_val
gc.collect()

                
#特征重要性
#plt.figure()
#lgb.plot_importance(model, max_num_features=50)                            
#plt.savefig('lgbclassifier_importance.jpg')               

feature_importances = pd.Series(model.feature_importance(), model.feature_name()).sort_values(ascending=True)
pd.DataFrame(feature_importances).to_csv('lgb_cl_feature_importances.csv')
plt.figure(figsize=(16,10))
feature_importances.plot(kind='barh', title='Feature Importances')
plt.xlabel('Feature Importance Score')
plt.savefig('lgbclassifier_importance.jpg')

print('train_lgb_model finishing...')



#########################################改变数据划分#################################################
#train-auc:0.916674      val-auc:0.860553 :(基本)   :实际：0.324557
#train-auc:0.92314       val-auc:0.846565 增加季度、月份、天数
#train-auc:0.911859      val-auc:0.831971 增加转化率


















##########################################################################################
#valid_0's auc: 0.895346
#valid_0's auc: 0.902844
#valid_0's auc: 0.903742


#[2000]  valid_0's auc: 0.932792  :实际：	0.312510(明显过拟合，需要调节参数，减少模型复杂度)
#[2000]  valid_0's auc: 0.933051















