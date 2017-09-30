# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 02:09:56 2017

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

#########################################################################################


###########################################################################################
train = train_data[train_data['计划起飞时间'] < parse('2017-1-1')]
train['temp_label'] = train['实际起飞时间'] - train['temp_计划起飞时间']
train['temp_label'] = train['temp_label'].apply(lambda x: 1 if x>10800 else 0) #时间截7200是两小时
train = train.drop(['实际起飞时间','实际到达时间'], axis=1)
train['航班是否取消'] = train['航班是否取消'].map({'正常':0,'取消':1})
train['航班是否取消'] = train['航班是否取消'] + train['temp_label']
train['航班是否取消'] = train['航班是否取消'].apply(lambda x: 1 if x>0 else 0)
train = train.drop(['temp_label','temp_计划起飞时间'], axis=1)


val = train_data[train_data['计划起飞时间'] >= parse('2017-1-1')]
val['temp_label'] = val['实际起飞时间'] - val['temp_计划起飞时间']
val['temp_label'] = val['temp_label'].apply(lambda x: 1 if x>10800 else 0) #时间截10800是三小时
val = val.drop(['实际起飞时间','实际到达时间'], axis=1)
val['航班是否取消'] = val['航班是否取消'].map({'正常':0,'取消':1})
val['航班是否取消'] = val['航班是否取消'] + val['temp_label']
val['航班是否取消'] = val['航班是否取消'].apply(lambda x: 1 if x>0 else 0)
val = val.drop(['temp_label','temp_计划起飞时间'], axis=1)



del train_data
gc.collect()

X_train = train.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间','航班是否取消'], axis=1)
X_train = X_train.drop(['飞行季度', '飞行月份'], axis=1)
y_train = train['航班是否取消']

X_val = val.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间','航班是否取消'], axis=1)
X_val = X_val.drop(['飞行季度', '飞行月份'], axis=1)
y_val = val['航班是否取消']

del train, val
gc.collect()

dtrain=xgb.DMatrix(X_train,label=y_train)
dval=xgb.DMatrix(X_val,label=y_val)

del X_train, y_train, X_val, y_val
gc.collect()

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc', #'logloss','auc','error'
	    'gamma':0.05, #0.05, 0.1, 0.2
#	    'min_child_weight':0.7, #0.03
	    'max_depth':6, #8,6,5
	    'lambda':1,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.1,
	    'tree_method':'exact',
	    'seed':2017,
	    'nthread':-1,
}

def Ap_scores(preds, dtrain):
    labels = dtrain.get_label()
    return 'Ap_scores', average_precision_score(labels, preds)



#模型训练
watchlist = [(dtrain,'train'),(dval,'val')]
model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=20)
#model = xgb.train(params,dtrain,num_boost_round=2000,evals=watchlist,early_stopping_rounds=50)
#model = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist,early_stopping_rounds=30, feval=Ap_scores)
model.save_model('xgbclassifier.model')


train_pre = model.predict(dtrain)
print('train_pr_auc：',Ap_scores(train_pre, dtrain))
val_pre = model.predict(dval)
print('val_pr_auc：',Ap_scores(val_pre, dval))


#特征重要性
#plt.figure()
#xgb.plot_importance(model)                            
#plt.savefig('xgbclassifier_importance.jpg')                

feature_importances = pd.Series(model.get_fscore()).sort_values(ascending=True)
pd.DataFrame(feature_importances).to_csv('xgb_cl_feature_importances.csv')
plt.figure(figsize=(16,10))
feature_importances.plot(kind='barh', title='Feature Importances')
plt.xlabel('Feature Importance Score')
plt.savefig('xgbclassifier_importance.jpg')
plt.show()

print('train_xgb_model finishing...')



#########################################改变数据划分#################################################
#auc:
#train-auc:0.916674      val-auc:0.860553 :(基本)
#train-auc:0.92314       val-auc:0.846565 增加季度、月份、天数
#train-auc:0.911859      val-auc:0.831971 增加转化率


#logloss:
#train-logloss:0.103657  val-logloss:0.104774 :(基本)
#train-logloss:0.104184  val-logloss:0.109751 :增加季度、月份、天数



#########################################改变数据划分（100）#################################################
#train-auc:0.829365      val-auc:0.787925 :(基本) :实际：0.491589
#train_pr_auc： ('Ap_scores', 0.55508248351003653)
#val_pr_auc： ('Ap_scores', 0.4934546612131463)





                                           
                                           
#########################################改变数据划分（2000）#################################################
#train-auc:0.87503       val-auc:0.80056  :(基本) :实际 0.510007
#train_pr_auc： ('Ap_scores', 0.63584627990934339)
#val_pr_auc： ('Ap_scores', 0.5134590000887802)





#######################################训练测试划分不同的数据##################################################
#[99]    train-auc:0.797092      val-auc:0.78801 :(基本)
#train_pr_auc： ('Ap_scores', 0.5203507312589627)
#val_pr_auc： ('Ap_scores', 0.48992939425622084)








































