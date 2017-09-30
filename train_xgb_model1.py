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

train_target = train_data['航班是否取消']
train_data = train_data.drop(['出发日期', '到达日期', '计划起飞时间','计划到达时间','航班是否取消'], axis=1)
train_data = train_data.drop(['飞行季度', '飞行月份'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_target, test_size=0.2, random_state=2017)

del train_target, train_data
gc.collect()

dtrain=xgb.DMatrix(X_train,label=y_train)
dval=xgb.DMatrix(X_val,label=y_val)

del X_train, y_train, X_val, y_val
gc.collect()


params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
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



######################################（原始特征改变label）####################################################
#train-auc:0.824134      val-auc:0.821156 :(基本) :实际：0.495242	

# 增加季度、月份、天数
# 增加转化率













##########################################################################################
#train-auc:0.888954      val-auc:0.886518
#train-auc:0.898196      val-auc:0.898099
#train-auc:0.898684      val-auc:0.895278 :实际：	0.478699
#train-auc:0.906593      val-auc:0.903557 增加季度、月份、天数
#全部数据的转化率出现泄漏
#一年计算转化率，一年训练模型
#train-auc:0.905426      val-auc:0.90342 :实际：	0.457758（数据泄露）


#[1999]  train-auc:0.950823      val-auc:0.934904 :实际：	0.476116
#[1999]  train-auc:0.951048      val-auc:0.935105  :实际：	0.479053
#[1999]  train-auc:0.958289      val-auc:0.943421  :实际：	0.466218






















