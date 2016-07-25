# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:13:45 2016

@author: mellon
TalkingData
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
                self.df_event=pd.read_csv(value)
            elif key=='app_events':
                self.df_app_event=pd.read_csv(value)
            elif key=='gender_age_train':
                self.df_group_train=pd.read_csv(value)            
            elif key=='gender_age_test':
                self.df_group_test=pd.read_csv(value)
            elif key=='phone_brand_device_model':
                self.df_device=pd.read_csv(value)
            elif key=='app_labels':
                self.df_app_label=pd.read_csv(value)
            elif key=='label_categories':
                self.df_label_category=pd.read_csv(value)  
    
    def mergedata(self,module):
#        device_id=self.df_group_train.device_id
#        gender=self.df_group_train.gender
#        age=self.df_group_train.age      
#        longitude=merge0.longitude
#        latitude=merge0.latitude              
#        is_installed=merge1.is_installed
#        is_active=merge1.is_active
#        category=merge3.category        
        if module=='train':
            merge0=pd.merge(self.df_group_train,self.df_event,on='device_id',how='inner')
            merge1=pd.merge(merge0,self.df_app_event,on='event_id',how='inner')
            merge2=pd.merge(merge1,self.df_app_label,on='app_id',how='inner')
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