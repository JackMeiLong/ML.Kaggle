# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:19:23 2016

@author: meil

"""
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

class Kobe(object):
    
   def __init__(self):
       self.trainset=[]
       self.trainlabel=[]
       self.testset=[]
       self.testlabel=[]
       
   def loaddataset(self,path,module):
       df=pd.read_csv(path)
       
       shot_made_flag=df['shot_made_flag']
       
       shot_id=df['shot_id']
       action_type=pd.get_dummies(df['action_type'])
       combined_shot_type=pd.get_dummies(df['combined_shot_type'])
       lat=df['lat']
       loc_x=df['loc_x']
       loc_y=df['loc_y']
       lon=df['lon']
       minutes=df['minutes_remaining']
       period=df['period']
       playoffs=df['playoffs']
       seconds=df['seconds_remaining']
       time_combined=pd.DataFrame({'time_combined':minutes.values*60+seconds.values})
       distance=df['shot_distance']
       shot_type=pd.get_dummies(df['shot_type'])
       shot_zone_area=pd.get_dummies(df['shot_zone_area'])
       shot_zone_basic=pd.get_dummies(df['shot_zone_basic'])
       shot_zone_range=pd.get_dummies(df['shot_zone_range'])
       opponent=pd.get_dummies(df['opponent'])
      
       dataset=pd.concat([shot_made_flag,shot_id,action_type,combined_shot_type,loc_x,loc_y,
                          time_combined,period,playoffs,distance,shot_type,shot_zone_area,
                          shot_zone_basic,shot_zone_range,opponent],axis=1)
                          
       if module=='train':
           train_df=dataset[np.isnan(shot_made_flag)==False]
           self.trainlabel=train_df['shot_made_flag']
           train_df=train_df.drop(['shot_made_flag','shot_id'],axis=1)
           self.trainset=train_df
           
       elif module=='test':
           test_df=dataset[np.isnan(shot_made_flag)==True]
           test_df=test_df.drop(['shot_made_flag'],axis=1)
           self.testset=test_df
    
   def train_LR(self):
      samples=self.trainset.values
      target=self.trainlabel.values
      classifier_LR=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
      classifier_LR.fit(samples,target)
      
      return classifier_LR
   
   def train_RF(self,samples,target,k):
        if samples is None or target is None:
            samples=self.trainset.values
            target=self.trainlabel.values
            
        classifier_RF=RandomForestClassifier(n_estimators=k)
        classifier_RF.fit(samples,target)        
        
        return classifier_RF
    
   def train_GBDT(self):
        samples=self.trainset.values
        target=self.trainlabel.values
        classifier_GB=GradientBoostingClassifier(n_estimators=1000)
        classifier_GB.fit(samples,target)
        
        return classifier_GB
        
   def evaluate(self,y_pred):
        y_true=self.trainlabel.values
        m=np.shape(y_true)[0]
        count=0
        for i in xrange(m):
            if y_pred[i]==y_true[i]:
                count=count+1
        
        accuracy=float(count)/m
            
        print(classification_report(y_true, y_pred))
        
        return accuracy
        
   def cross_validation(self,clf):
       samples=self.trainset.values
       target=self.trainlabel.values
       scores=cross_val_score(clf,samples,target,cv=10,scoring='accuracy')
       
       return scores
       
   def feature_select(self,clf,k):
       selector=RFE(clf,k,step=1)
       samples=self.trainset.values
       target=self.trainlabel.values
       selector=selector.fit(samples,target)
       return selector
       
   def dimension_reduce(self,k):   
       pca=PCA(n_components=k)
       samples=self.trainset.values
       pca.fit(samples)
       
       return pca
       
       
   def tocsv(self,y_pred):
        submit=pd.DataFrame({'shot_id':self.testset['shot_id'].values,
                            'shot_made_flag':y_pred})    
        submit.to_csv('sample_submission.csv',index=False) 
   
   def optimize(self):
       num_rf=np.linspace(10,90,5)
       dim_pca=np.linspace(10,90,5)
       
       m=np.shape(dim_pca)[0]
       n=np.shape(num_rf)[0]
       
       accuracy=np.zeros((m,n))
#       label_test=np.zeros((m*n,1))
       
       for i in range(len(dim_pca)):
           pca=self.dimension_reduce(int(dim_pca[i]))
           trainset_new=pca.transform(self.trainset.values)
#           inX_new=pca.transform(self.testset.values[:,1:])
           for j in range(len(num_rf)):
               classifier_RF=kobe.train_RF(trainset_new,trainlabel,int(num_rf[j]))
               y_train_pred_RF=classifier_RF.predict(trainset_new)
               print 'Dimension: {0} ; RF: {1}'.format(int(dim_pca[i]),int(num_rf[j]))
               accuracy_train_RF=kobe.evaluate(y_train_pred_RF)
#               y_test_RF=classifier_RF.predict_proba(inX_new)
               accuracy[i][j]=accuracy_train_RF
        
       return accuracy
               

#==============================================================================
# Main
#==============================================================================
time_start=time.time()

print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

path='data.csv'

kobe=Kobe()

kobe.loaddataset(path,'train')
kobe.loaddataset(path,'test')

trainset=kobe.trainset.values
trainlabel=kobe.trainlabel.values
testset=kobe.testset.values
inX=testset[:,1:]

#==============================================================================
# Preprocess
#==============================================================================

print 'Decomposition:PCA'

k=10
pca=kobe.dimension_reduce(k)

trainset_new=pca.transform(trainset)
inX_new=pca.transform(inX)

#==============================================================================
# Logistic Regression
#==============================================================================
#classifier_LR=kobe.train_LR()
#
#print 'LR-Train: Cross-Validation'
#
#scores_LR= kobe.cross_validation(classifier_LR)
#print scores_LR
#
#y_train_pred_LR=classifier_LR.predict(trainset)
#
#print 'LR-Train: Precision & Recall'
#accuracy_train_LR=kobe.evaluate(y_train_pred_LR)
#
#print 'LR-Train: Accuracy'
#print accuracy_train_LR
#
#y_test_LR=classifier_LR.predict_proba(inX)

#kobe.tocsv(y_test_LR[:,1])

#==============================================================================
# Random Forest
#==============================================================================

classifier_RF=kobe.train_RF(trainset_new,trainlabel,10)

#print 'RF-Train: Cross-Validation'
#print kobe.cross_validation(classifier_RF)

y_train_pred_RF=classifier_RF.predict(trainset_new)

print 'RF-Train: Precision & Recall'
accuracy_train_RF=kobe.evaluate(y_train_pred_RF)

print 'RF-Train: Accuracy'
print accuracy_train_RF

y_test_RF=classifier_RF.predict_proba(inX_new)

kobe.tocsv(y_test_RF[:,1])


#accuracy=kobe.optimize()


#==============================================================================
# GBDT
#==============================================================================

#classifier_GBDT=kobe.train_GBDT()

#print 'RF-GBDT: Cross-Validation'
#print kobe.cross_validation(classifier_GBDT)

#y_train_pred_GBDT=classifier_GBDT.predict(trainset)
#
#print 'GBDT-Train: Precision & Recall'
#accuracy_train_GBDT=kobe.evaluate(y_train_pred_GBDT)
#
#print 'GBDT-Train: Accuracy'
#print accuracy_train_GBDT
#
#y_test_GBDT=classifier_GBDT.predict_proba(inX)

#kobe.tocsv(y_test_GBDT[:,1])


print '<----------------------------------------------->'
print datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print 'Time During'     
time_end=time.time()
time_during=time_end-time_start
print time_during

     
      