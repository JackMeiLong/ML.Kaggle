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
        self.df_device=[]
        self.df_app_label=[]
        self.df_label_category=[]
        self.df_group_test=[]
        self.train=[]
        self.label=[]
        self.test=[]
    
    def importdata(self,paths_dict):
        
        for (key,value) in paths_dict:
            "df_{0}".format(key)=pd.read_csv(value)
    






td=TalkingData()

paths_dict={'events':'events.csv','app_events':'app_events.csv','gender_age_train':'gender_age_train.csv',
            'gender_age_test':'gender_age_test.csv','phone_brand_device_model':'phone_brand_device_model.csv',
            'app_labels':'app_labels.csv','label_categories':'label_categories.csv'}

td.importdata(paths_dict)