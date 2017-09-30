# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:48:14 2017

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

def flight_pro():
    train_flight = pd.read_csv(r'train/flight.csv', nrows=None, encoding='gbk')
    test_flight = pd.read_csv(r'test/flight.csv', nrows=None, encoding='gbk')
    train_flight['实际起飞时间'] = train_flight['实际起飞时间'].fillna(train_flight['计划起飞时间'].min())
    train_flight['实际到达时间'] = train_flight['实际到达时间'].fillna(train_flight['计划到达时间'].min())
    
#    train_flight['temp_label'] = train_flight['实际起飞时间'] - train_flight['计划起飞时间']
#    train_flight['temp_label'] = train_flight['temp_label'].apply(lambda x: 1 if x>10800 else 0) #时间截10800是三小时
    #对计划起飞时间、计划到达时间进行转换(timestamp转datetime:)
    '''>>> datetime.datetime.fromtimestamp(1437123812.0)
        datetime.datetime(2015, 7, 17, 17, 3, 32)
        >>> print datetime.datetime.fromtimestamp(1437123812.0)
        2015-07-17 17:03:32
        >>> import time
        >>> now=datetime.datetime.now()
        >>> time.mktime(now.timetuple())
        1437123812.0'''
    train_flight['temp_计划起飞时间'] = train_flight['计划起飞时间']  #临时需要的字段备份一个

    train_flight['计划起飞时间'] = train_flight['计划起飞时间'].apply(lambda x: datetime.fromtimestamp(x))
    train_flight['计划到达时间'] = train_flight['计划到达时间'].apply(lambda x: datetime.fromtimestamp(x))

    test_flight['计划起飞时间'] = test_flight['计划起飞时间'].apply(lambda x: datetime.fromtimestamp(x))
    test_flight['计划到达时间'] = test_flight['计划到达时间'].apply(lambda x: datetime.fromtimestamp(x))

    train_flight['出发日期'] = train_flight['计划起飞时间'].apply(lambda x: x.date())
    train_flight['到达日期'] = train_flight['计划到达时间'].apply(lambda x: x.date())
    
    test_flight['出发日期'] = test_flight['计划起飞时间'].apply(lambda x: x.date())
    test_flight['到达日期'] = test_flight['计划到达时间'].apply(lambda x: x.date())
    
    train_flight['出发小时'] = train_flight['计划起飞时间'].apply(lambda x: x.hour)
    train_flight['到达小时'] = train_flight['计划到达时间'].apply(lambda x: x.hour)
    
    test_flight['出发小时'] = test_flight['计划起飞时间'].apply(lambda x: x.hour)
    test_flight['到达小时'] = test_flight['计划到达时间'].apply(lambda x: x.hour)
    
    train_flight['飞行季度'] = train_flight['计划到达时间'].apply(lambda x: x.quarter)
    test_flight['飞行季度'] = test_flight['计划起飞时间'].apply(lambda x: x.quarter)
    
    train_flight['飞行月份'] = train_flight['计划到达时间'].apply(lambda x: x.month)
    test_flight['飞行月份'] = test_flight['计划起飞时间'].apply(lambda x: x.month)
    
    
    train_flight.to_csv(r'data_pro/train_flight.csv', index=False)
    test_flight.to_csv(r'data_pro/test_flight.csv', index=False)
    
    print(train_flight.shape, test_flight.shape)
    print('flight_pro finish......')
    return train_flight, test_flight

def out_weather_pro():
    train_weather = pd.read_csv(r'train/weather.csv', nrows=None)
    train_weather = train_weather.drop('Unnamed: 5', axis=1)
    train_weather = train_weather.drop_duplicates(['城市','日期'])
    test_weather = pd.read_excel(r'test/weather.xlsx', nrows=None)
    test_weather = test_weather.drop_duplicates(['城市','日期'])
    
    train_weather.columns = ['出发'+x for x in train_weather.columns]
    test_weather.columns = ['出发'+x for x in test_weather.columns]
    
    train_weather.to_csv(r'data_pro/train_weather_out.csv', index=False)
    test_weather.to_csv(r'data_pro/test_weather_out.csv', index=False)
    
    print(train_weather.shape, test_weather.shape)
    print('out_weather_pro finish......')
    return train_weather, test_weather

def in_weather_pro():
    train_weather = pd.read_csv(r'train/weather.csv', nrows=None)
    train_weather = train_weather.drop('Unnamed: 5', axis=1)
    train_weather = train_weather.drop_duplicates(['城市','日期'])
    test_weather = pd.read_excel(r'test/weather.xlsx', nrows=None)
    test_weather = test_weather.drop_duplicates(['城市','日期'])
    
    train_weather.columns = ['到达'+x for x in train_weather.columns]
    test_weather.columns = ['到达'+x for x in test_weather.columns]
    
    train_weather.to_csv(r'data_pro/train_weather_in.csv', index=False)
    test_weather.to_csv(r'data_pro/test_weather_in.csv', index=False)

    print(train_weather.shape, test_weather.shape)    
    print('out_weather_pro finish......')
    return train_weather, test_weather

def out_city_pro():
    train_city = pd.read_excel(r'train/city.xlsx', nrows=None)
    test_city = pd.read_excel(r'test/city.xlsx', nrows=None)
    train_city.columns = ['机场','城市']
    test_city.columns = ['机场','城市']
    train_city.columns = ['出发'+x for x in train_city.columns]
    test_city.columns = ['出发'+x for x in test_city.columns]
    
    train_city.to_csv(r'data_pro/train_city_out.csv', index=False)
    test_city.to_csv(r'data_pro/test_city_out.csv', index=False)
        
    print(train_city.shape, test_city.shape)
    print('out_city_pro finish......')
    return train_city, test_city
    
def in_city_pro():
    train_city = pd.read_excel(r'train/city.xlsx', nrows=None)
    test_city = pd.read_excel(r'test/city.xlsx', nrows=None)
    train_city.columns = ['机场','城市']
    test_city.columns = ['机场','城市']
    train_city.columns = ['到达'+x for x in train_city.columns]
    test_city.columns = ['到达'+x for x in test_city.columns]
    
    train_city.to_csv(r'data_pro/train_city_in.csv', index=False)
    test_city.to_csv(r'data_pro/test_city_in.csv', index=False)
    
    print(train_city.shape, test_city.shape)
    print('in_city_pro finish......')
    return train_city, test_city


def city_weather_pro():
    train_weather_out, test_weather_out = out_weather_pro()
    train_weather_in, test_weather_in = in_weather_pro()
    train_city_out, test_city_out = out_city_pro()
    train_city_in, test_city_in = in_city_pro()
    
    train_city_weather_out = pd.merge(train_city_out, train_weather_out, on='出发城市', how='left')
    train_city_weather_in = pd.merge(train_city_in, train_weather_in, on='到达城市', how='left')
    test_city_weather_out = pd.merge(test_city_out, test_weather_out, on='出发城市', how='left')
    test_city_weather_in = pd.merge(test_city_in, test_weather_in, on='到达城市', how='left')
    
    train_city_weather_out.to_csv(r'data_pro/train_city_weather_out.csv', index=False)
    train_city_weather_in.to_csv(r'data_pro/train_city_weather_in.csv', index=False)
    test_city_weather_out.to_csv(r'data_pro/test_city_weather_out.csv', index=False)
    test_city_weather_in.to_csv(r'data_pro/test_city_weather_in.csv', index=False)
    
    print(train_city_weather_out.shape, train_city_weather_in.shape, test_city_weather_out.shape, test_city_weather_in.shape)
    print('city_weather_pro finish......')
    return train_city_weather_out, train_city_weather_in, test_city_weather_out, test_city_weather_in
    
    
def Informant_pro():    
    train_Informant = pd.read_excel(r'train/Informant.xlsx', nrows=None)
    train_Informant = train_Informant.dropna(axis=0)
    train_Informant['Informant_label'] = train_Informant['特情内容'].apply(lambda x: 0 if (x == '跑道开放') | (x == '机场开放') else 1)
    train_Informant['开始时间'] = pd.to_datetime(train_Informant['开始时间'])
    train_Informant['结束时间'] = pd.to_datetime(train_Informant['结束时间'])
    
    temp_train_Informant = pd.DataFrame(columns=['机场','时间','Informant_label'])
    for i in range(train_Informant.shape[0]):
        temp = pd.DataFrame(columns=['机场','时间','Informant_label'])
        temp['时间'] = pd.date_range(start=train_Informant.iloc[i]['开始时间'], end=train_Informant.iloc[i]['结束时间'], freq='1H')
        temp['机场'] = train_Informant.iloc[i]['特情机场']
        temp['Informant_label'] = train_Informant.iloc[i]['Informant_label']
        temp_train_Informant = pd.concat([temp_train_Informant, temp])


    test_Informant = pd.read_excel(r'test/Informant.xlsx', nrows=None)    
    test_Informant = test_Informant.dropna(axis=0)
    test_Informant['Informant_label'] = test_Informant['特情内容'].apply(lambda x: 0 if (x == '跑道开放') | (x == '机场开放') else 1)
    test_Informant['开始时间'] = pd.to_datetime(test_Informant['开始时间'])
    test_Informant['结束时间'] = pd.to_datetime(test_Informant['结束时间'])
    
    temp_test_Informant = pd.DataFrame(columns=['机场','时间','Informant_label'])
    for i in range(test_Informant.shape[0]):
        temp = pd.DataFrame(columns=['机场','时间','Informant_label'])
        temp['时间'] = pd.date_range(start=test_Informant.iloc[i]['开始时间'], end=test_Informant.iloc[i]['结束时间'], freq='1H')
        temp['机场'] = test_Informant.iloc[i]['特情机场']
        temp['Informant_label'] = test_Informant.iloc[i]['Informant_label']
        temp_test_Informant = pd.concat([temp_test_Informant, temp])

    temp_train_Informant.to_csv(r'data_pro/temp_train_Informant.csv', index=False)
    temp_test_Informant.to_csv(r'data_pro/temp_test_Informant.csv', index=False)

    print(temp_train_Informant.shape, temp_test_Informant.shape)
    print('Informant_pro finish......')
    return temp_train_Informant, temp_test_Informant


def out_Informant_pro():  
    train_Informant = pd.read_csv(r'data_pro/temp_train_Informant.csv',\
                              nrows=None, encoding='gbk')  
    test_Informant = pd.read_csv(r'data_pro/temp_test_Informant.csv', \
                                 nrows=None, encoding='gbk')   
    train_Informant['时间'] = pd.to_datetime(train_Informant['时间'])
    test_Informant['时间'] = pd.to_datetime(test_Informant['时间'])
    train_Informant.columns = ['出发'+x for x in train_Informant.columns]
    test_Informant.columns = ['出发'+x for x in test_Informant.columns]
    train_Informant['出发日期'] = train_Informant['出发时间'].apply(lambda x: x.date())
    test_Informant['出发日期'] = test_Informant['出发时间'].apply(lambda x: x.date())
    train_Informant['出发小时'] = train_Informant['出发时间'].apply(lambda x: x.hour)
    test_Informant['出发小时'] = test_Informant['出发时间'].apply(lambda x: x.hour)
    train_Informant = train_Informant.drop('出发时间', axis=1)
    test_Informant = test_Informant.drop('出发时间', axis=1)
    
    train_Informant.to_csv(r'data_pro/train_Informant_out.csv', index=False)
    test_Informant.to_csv(r'data_pro/test_Informant_out.csv', index=False)
        
    print(train_Informant.shape, test_Informant.shape)
    print('out_Informant_pro finish......')
    return train_Informant, test_Informant
    

def in_Informant_pro():  
    train_Informant = pd.read_csv(r'data_pro/temp_train_Informant.csv',\
                              nrows=None, encoding='gbk')  
    test_Informant = pd.read_csv(r'data_pro/temp_test_Informant.csv', \
                                 nrows=None, encoding='gbk')   
    train_Informant['时间'] = pd.to_datetime(train_Informant['时间'])
    test_Informant['时间'] = pd.to_datetime(test_Informant['时间'])
    train_Informant.columns = ['到达'+x for x in train_Informant.columns]
    test_Informant.columns = ['到达'+x for x in test_Informant.columns]
    train_Informant['到达日期'] = train_Informant['到达时间'].apply(lambda x: x.date())
    test_Informant['到达日期'] = test_Informant['到达时间'].apply(lambda x: x.date())
    train_Informant['到达小时'] = train_Informant['到达时间'].apply(lambda x: x.hour)
    test_Informant['到达小时'] = test_Informant['到达时间'].apply(lambda x: x.hour)
    train_Informant = train_Informant.drop('到达时间', axis=1)
    test_Informant = test_Informant.drop('到达时间', axis=1)
    
    train_Informant.to_csv(r'data_pro/train_Informant_in.csv', index=False)
    test_Informant.to_csv(r'data_pro/test_Informant_in.csv', index=False)
        
    print(train_Informant.shape, test_Informant.shape)
    print('in_Informant_pro finish......')
    return train_Informant, test_Informant






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
    flight_pro()
#    out_weather_pro()
#    in_weather_pro()
#    out_city_pro()
#    in_city_pro()
#    city_weather_pro()
#    Informant_pro()
#    out_Informant_pro()
#    in_Informant_pro()
    marge_data()
    
    
    
    
    
    
    print('finish......')
    
    
    
    
    
    
    
    
    
    











