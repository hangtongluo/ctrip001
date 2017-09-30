# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:24:19 2017

@author: Administrator
"""

import gc
from collections import Counter
from datetime import datetime 
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import LabelEncoderTool

def marge_data():
    '''出发机场, 到达机场, 航班编号, 航班是否取消, 出发日期, 到达日期, 出发城市, 出发天气, 
    出发最低气温, 出发最高气温, 到达城市, 到达天气, 到达最低气温, 到达最高气温'''
    
    train_flight = pd.read_csv(r'data_pro/train_flight.csv', nrows=None, encoding='gbk')
#    train_flight = train_flight.drop(['实际起飞时间','实际到达时间'], axis=1)
    train_city_weather_out = pd.read_csv(r'data_pro/train_city_weather_out.csv', nrows=None, encoding='gbk')
    train_city_weather_in = pd.read_csv(r'data_pro/train_city_weather_in.csv', nrows=None, encoding='gbk')
    train_Informant_out = pd.read_csv(r'data_pro/train_Informant_out.csv', nrows=None, encoding='gbk')
#    train_Informant_out = train_Informant_out.drop_duplicates()
    train_Informant_out = train_Informant_out.drop_duplicates(['出发机场','出发日期'])
    train_Informant_out = train_Informant_out.drop('出发小时', axis=1)
    train_Informant_in = pd.read_csv(r'data_pro/train_Informant_in.csv', nrows=None, encoding='gbk')
#    train_Informant_in = train_Informant_in.drop_duplicates()
    train_Informant_in = train_Informant_in.drop_duplicates(['到达机场','到达日期'])
    train_Informant_in = train_Informant_in.drop('到达小时', axis=1)

    train_data = pd.merge(train_flight, train_city_weather_out, on=['出发机场','出发日期'], how='left')
    train_data = pd.merge(train_data, train_city_weather_in, on=['到达机场','到达日期'], how='left')
#    train_data = pd.merge(train_data, train_Informant_out, on=['出发机场','出发日期','出发小时'], how='left')
#    train_data = pd.merge(train_data, train_Informant_in, on=['到达机场','到达日期','到达小时'], how='left')
    train_data = pd.merge(train_data, train_Informant_out, on=['出发机场','出发日期'], how='left')
    train_data = pd.merge(train_data, train_Informant_in, on=['到达机场','到达日期'], how='left')
    train_data['出发Informant_label'] = train_data['出发Informant_label'].fillna(0)
    train_data['到达Informant_label'] = train_data['到达Informant_label'].fillna(0)
#    train_data['出发Informant_label'] = train_data['出发Informant_label'].fillna(-1)
#    train_data['到达Informant_label'] = train_data['到达Informant_label'].fillna(-1)
    
    print(train_flight.shape, train_city_weather_out.shape, train_city_weather_in.shape, train_data.shape)
    print(train_Informant_out.shape, train_Informant_in.shape)
    del train_flight, train_city_weather_out, train_city_weather_in
    del train_Informant_out, train_Informant_in
    gc.collect()
    
    
    test_flight = pd.read_csv(r'data_pro/test_flight.csv', nrows=None, encoding='gbk')
    test_city_weather_out = pd.read_csv(r'data_pro/test_city_weather_out.csv', nrows=None, encoding='gbk')
    test_city_weather_in = pd.read_csv(r'data_pro/test_city_weather_in.csv', nrows=None, encoding='gbk')
    test_Informant_out = pd.read_csv(r'data_pro/test_Informant_out.csv', nrows=None, encoding='gbk')    
#    test_Informant_out = test_Informant_out.drop_duplicates()
    test_Informant_out = test_Informant_out.drop_duplicates(['出发机场','出发日期'])
    test_Informant_out = test_Informant_out.drop('出发小时', axis=1)
    
    test_Informant_in = pd.read_csv(r'data_pro/test_Informant_in.csv', nrows=None, encoding='gbk')
#    test_Informant_in = test_Informant_in.drop_duplicates()
    test_Informant_in = test_Informant_in.drop_duplicates(['到达机场','到达日期'])
    test_Informant_in = test_Informant_in.drop('到达小时', axis=1)
    
    test_data = pd.merge(test_flight, test_city_weather_out, on=['出发机场','出发日期'], how='left')
    test_data = pd.merge(test_data, test_city_weather_in, on=['到达机场','到达日期'], how='left')
#    test_data = pd.merge(test_data, test_Informant_out, on=['出发机场','出发日期','出发小时'], how='left')
#    test_data = pd.merge(test_data, test_Informant_in, on=['到达机场','到达日期','到达小时'], how='left')
    test_data = pd.merge(test_data, test_Informant_out, on=['出发机场','出发日期'], how='left')
    test_data = pd.merge(test_data, test_Informant_in, on=['到达机场','到达日期'], how='left')
    test_data['出发Informant_label'] = test_data['出发Informant_label'].fillna(0)
    test_data['到达Informant_label'] = test_data['到达Informant_label'].fillna(0)
#    test_data['出发Informant_label'] = test_data['出发Informant_label'].fillna(-1)
#    test_data['到达Informant_label'] = test_data['到达Informant_label'].fillna(-1)
    
    print(test_flight.shape, test_city_weather_out.shape, test_city_weather_in.shape, test_data.shape)
    print(test_Informant_out.shape, test_Informant_in.shape)
    del test_flight, test_city_weather_out, test_city_weather_in
    del test_Informant_out, test_Informant_in
    gc.collect()
    
    train_data.to_csv(r'features/train_data_look_features.csv', index=False) #中间数据进行观察
    
    usecols = ['出发机场', '到达机场', '航班编号', '出发城市', '出发天气', \
               '出发最低气温', '出发最高气温', '到达城市', '到达天气', '到达最低气温', '到达最高气温']
    train_data, test_data = LabelEncoderTool(train_data, test_data, usecols)
    
#    train_data['航班是否取消'] = train_data['航班是否取消'].map({'正常':0,'取消':1})
#    train_data['航班是否取消'] = train_data['航班是否取消'] + train_data['temp_label']
#    train_data['航班是否取消'] = train_data['航班是否取消'].apply(lambda x: 1 if x>0 else 0)
#    train_data = train_data.drop('temp_label', axis=1)
    
    train_data.to_csv(r'features/train_data.csv', index=False)
    test_data.to_csv(r'features/test_data.csv', index=False)
    
    print(train_data.shape, test_data.shape)
    print('city_weather_marge finish......')
    return train_data, test_data












if __name__ == '__main__':
    marge_data()
#    make_features1()
    
    
    
    
    
    
    print('finish......')









