# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:13:45 2016

@author: mellon
TalkingData

https://www.kaggle.com/xiaoml/talkingdata-mobile-user-demographics/bag-of-app-id-python-2-27392/code

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
                self.df_device=pd.read_csv(value,dtype={'device_id': np.str}).drop(['Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8'],axis=1)
            elif key=='app_labels':
                self.df_app_label=pd.read_csv(value,dtype={'app_id': np.str,'label_id': np.str})
            elif key=='label_categories':
                self.df_label_category=pd.read_csv(value,dtype={'label_id': np.str})  
    
    def mergedata(self,module):
        
        if module=='train':
            self.df_group_train=self.df_group_train.drop(['gender','age'],axis=1)
            merge0=pd.merge(self.df_group_train,self.df_event,on='device_id',how='left')
        elif module=='test':
            merge0=pd.merge(self.df_group_test,self.df_event,on='device_id',how='left')
    
        merge1=pd.merge(merge0,self.df_app_event,on='event_id',how='left')
        merge1=merge1.drop(['event_id'],axis=1)
        merge2=pd.merge(merge1,self.df_app_label,on='app_id',how='left')
        merge2=merge2.drop(['app_id'],axis=1)
        merge3=pd.merge(merge2,self.df_label_category,on='label_id',how='left')
        merge4=pd.merge(merge3,self.df_device,on='device_id',how='left')
        
        if module=='train':
            train=merge4
            label=self.df_group_train.group
            self.train=train.drop(['device_id','label_id','group'],axis=1)
            self.label=label
        elif module=='test':    
            test=merge4
            self.test=test.drop(['device_id','label_id'],axis=1)
    

td=TalkingData()

paths_dict={'events':'events.csv','app_events':'app_events.csv','gender_age_train':'gender_age_train.csv',
            'gender_age_test':'gender_age_test.csv','phone_brand_device_model':'phone_brand_device_model.csv',
            'app_labels':'app_labels.csv','label_categories':'label_categories.csv'}

#td.importdata(paths_dict)
#td.mergedata('train')
#td.mergedata('test')

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("app_events.csv", dtype={'device_id': np.str})
# remove duplicates(app_id)
app_ev = app_ev.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("events.csv", dtype={'device_id': np.str})
events["app_id"] = events["event_id"].map(app_ev)

events = events.dropna()

del app_ev

events = events[["device_id", "app_id"]]

# remove duplicates(app_id)
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")

train = pd.read_csv("gender_age_train.csv",
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)

test = pd.read_csv("gender_age_test.csv",
                   dtype={'device_id': np.str})
test["group"] = np.nan


split_len = len(train)

Df = pd.concat((train, test), axis=0, ignore_index=True)

Df = pd.merge(Df, pbd, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model"] = Df["device_model"].apply(lambda x: "device_model:" + str(x))
