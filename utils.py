# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 01:13:34 2017

@author: Administrator
"""

import gc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb



#############################basis process functions###################################
#变量编码函数
def LabelEncoderTool(trainDF, testDF, usecol):
    alldata = pd.concat([trainDF, testDF])
    for col in usecol:
        le = LabelEncoder()
        le.fit(alldata[col].astype('str'))
        trainDF[col] = le.transform(trainDF[col].astype('str'))
        testDF[col] = le.transform(testDF[col].astype('str'))
    return trainDF, testDF



########################################################################################################
#定义转化率函数
def rateSmoothing(df):
    count = df['航班是否取消'].count() + 10  #简单的平滑
    summ = df['航班是否取消'].sum()
    df['rate'] = summ / count
    return df

#定义转化率函数
def conversionRate(DF, usecols):
    new_colname = '_'.join(usecols[:-1])+'_rate'
    ratefeatures = DF[usecols].groupby(usecols[:-1]).apply(rateSmoothing).rename(columns={'rate':new_colname})
    ratefeatures = ratefeatures.drop('航班是否取消', axis=1)
    return ratefeatures


########################################################################################################




















