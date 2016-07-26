# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:53:08 2016

@author: meil
"""

import pandas as pd
import numpy as np

class TalkingData(object):
    
    def __init__(self):
        self.df_event=[]
        self.df_app_event=[]
        self.df_group_train=[]
        self.df_group_test=[]
        self.df_device=[]
        self.df_app_label=[]
        self.df_label_category=[]       
        self.train=[]
        self.label=[]
        self.test=[]
    
    def importdata(self,paths_dict):
        
        for (key,value) in paths_dict.iteritems():
            if key=='events':
                self.df_event=pd.read_csv(value,dtype={'device_id': np.str,'event_id': np.str})
            elif key=='app_events':
                self.df_app_event=pd.read_csv(value,dtype={'is_installed':np.int,'is_active':np.int,'event_id': np.str,'app_id': np.str})
            elif key=='gender_age_train':
                self.df_group_train=pd.read_csv(value,dtype={'device_id': np.str})            
            elif key=='gender_age_test':
                self.df_group_test=pd.read_csv(value,dtype={'device_id': np.str})
            elif key=='phone_brand_device_model':
                self.df_device=pd.read_csv(value,dtype={'device_id': np.str})
            elif key=='app_labels':
                self.df_app_label=pd.read_csv(value,dtype={'app_id': np.str,'label_id': np.str})
            elif key=='label_categories':
                self.df_label_category=pd.read_csv(value,dtype={'label_id': np.str})  
    
    def mergedata(self,module):
        
        if module=='train':
            merge0=pd.merge(self.df_group_train,self.df_event,on='device_id',how='inner')
            merge1=pd.merge(merge0,self.df_app_event,on='event_id',how='inner')
            merge1=merge1.drop(['event_id'],axis=1)
            merge2=pd.merge(merge1,self.df_app_label,on='app_id',how='inner')
            merge2=merge2.drop(['app_id','label_id'],axis=1)
            merge3=pd.merge(merge2,self.df_label_category,on='label_id',how='inner')
            train=merge3
            label=self.df_group_train.group
            self.train=train.drop(['device_id','gender','age','event_id','app_id','label_id'],axis=1)
            self.label=label
        elif module=='test':
            merge0=pd.merge(self.df_group_test,self.df_event,on='device_id',how='inner')
            merge1=pd.merge(merge0,self.df_app_event,on='event_id',how='inner')
            merge2=pd.merge(merge1,self.df_app_label,on='app_id',how='inner')
            merge3=pd.merge(merge2,self.df_label_category,on='label_id',how='inner')
            test=merge3
            self.test=test.drop(['device_id','event_id','app_id','label_id'],axis=1)
    

td=TalkingData()

paths_dict={'events':'events.csv','app_events':'app_events.csv','gender_age_train':'gender_age_train.csv',
            'gender_age_test':'gender_age_test.csv','phone_brand_device_model':'phone_brand_device_model.csv',
            'app_labels':'app_labels.csv','label_categories':'label_categories.csv'}

td.importdata(paths_dict)
td.mergedata('train')
td.mergedata('test')


#print('Read events...')
#events = pd.read_csv("events.csv", dtype={'device_id': np.str})
#events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
#events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
#
#    # Phone brand
#print('Read brands...')
#pbd = pd.read_csv("phone_brand_device_model.csv", dtype={'device_id': np.str})
#pbd.drop_duplicates('device_id', keep='first', inplace=True)
#pbd = map_column(pbd, 'phone_brand')
#pbd = map_column(pbd, 'device_model')
#
#    # Train
#print('Read train...')
#train = pd.read_csv("gender_age_train.csv", dtype={'device_id': np.str})
#train = map_column(train, 'group')
#train = train.drop(['age'], axis=1)
#train = train.drop(['gender'], axis=1)
#train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
#train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
#train.fillna(-1, inplace=True)
#
#    # Test
#print('Read test...')
#test = pd.read_csv("gender_age_test.csv", dtype={'device_id': np.str})
#test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
#test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
#test.fillna(-1, inplace=True)
#
#    # Features
#features = list(test.columns.values)
#features.remove('device_id')

