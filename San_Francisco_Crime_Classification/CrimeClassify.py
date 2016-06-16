# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:15:38 2016

@author: meil

San_Francisco_Crime_Classification

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time
import datetime

class CrimeClassify(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.testset=[]
        self.testlabel=[]
        self.category=[]
        self.origin=[]
        
    def loaddataset(self,path,module):
        df=pd.read_csv(path)
        
        subdf=df[['Dates','DayOfWeek','PdDistrict','Address','X','Y']]       
        DayOfWeek=subdf['DayOfWeek']
        dummies_Day=pd.get_dummies(DayOfWeek,prefix='DayOfWeek')
        PdDistrict=subdf['PdDistrict']     
        dummies_PdDistrict=pd.get_dummies(PdDistrict,prefix='PdDistrict')
#       Address=subdf['Address']
        X=subdf['X']
        Y=subdf['Y']
        
        if module=='train':
            train_subdf=df[['Category','Descript','Resolution']]
            Category=train_subdf['Category']
            Descript=train_subdf['Descript']
#           Resolution=train_subdf['Resolution']
            self.category=Category.unique().tolist()
            self.trainset=pd.concat([dummies_Day,dummies_PdDistrict,X,Y],axis=1)
            self.trainlabel=Category
            self.origin=df
        elif module=='test':
            test_subdf=df[['Id']]
            ID=test_subdf['Id']
            self.testset=pd.concat([ID,dummies_Day,dummies_PdDistrict,X,Y],axis=1)
        
    def train_LR(self):
        samples=self.trainset.values
        target=self.trainlabel.values
        classifier_LR=LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        classifier_LR.fit(samples,target)
        
        return classifier_LR
        
    def train_RF(self):
        samples=self.trainset.values
        target=self.trainlabel.values
        classifier_RF=RandomForestClassifier(n_estimators=1000)
        classifier_RF.fit(samples,target)        
        
        return classifier_RF
    
    def train_KNN(self):
        classifier_KNN=KNeighborsClassifier(n_neighbors=5,algorithm ='auto')
        samples=self.trainset.values
        target=self.trainlabel.values
        classifier_KNN.fit(samples,target)
        
        return classifier_KNN
        
    def evaluate(self,y_pred):
        y_true=self.trainlabel.values
        m=np.shape(y_true)[0]
        count=0
        for i in xrange(m):
            if y_pred[i]==y_true[i]:
                count=count+1
        
        accuracy=float(count)/m
        
        target_names = self.category
        
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return accuracy
        
    def tocsv(self,y_pred):
        submit=pd.DataFrame({'Id':self.testset.Id})
        for temp in self.category:
            submit[temp]=np.where(y_pred==temp,1,0)
            
        submit.to_csv('sampleSubmission.csv',index=False)
        
#==============================================================================
# Main
#==============================================================================
time_start=time.time()

print str(datetime.datetime.now())

CC=CrimeClassify()

train_path="train.csv"

CC.loaddataset(train_path,'train')

test_path='test.csv'

CC.loaddataset(test_path,'test')

trainset=CC.trainset.values
trainlabel=CC.trainlabel.values

testset=CC.testset.values

inX=testset[:,1:]

#==============================================================================
# Logistic Regression
#==============================================================================

classifier_LR=CC.train_LR()

print 'Precision-Recall-LR-Train'

y_train_pred_LR=classifier_LR.predict(trainset)

accuracy_train_LR=CC.evaluate(y_train_pred_LR)

y_test_LR=classifier_LR.predict_proba(inX)

print 'accuracy_train_LR'

print accuracy_train_LR

#CC.tocsv(y_test_LR)

#print 'Cross-Validation : Logistic Regression'

#scores_LR=titanic.cross_validation(classifier_LR,5)

#==============================================================================
# Random Forest
#==============================================================================

#classifier_RF=CC.train_RF()
#
#print 'Precision-Recall-RF-Train'
#
##y_train_pred_RF=classifier_RF.predict(trainset)
##
##accuracy_train_RF=CC.evaluate(y_train_pred_RF)
#
#y_test_RF=classifier_RF.predict(inX)
#
#CC.tocsv(y_test_RF)

#==============================================================================
# KNN
#==============================================================================

#classifier_KNN=CC.train_KNN()
#
#print 'Precision-Recall-KNN-Train'
#
#y_train_pred_KNN=classifier_KNN.predict(trainset)
#
#y_test_KNN=classifier_KNN.predict(inX)
#
##print 'Cross-Validation : Logistic Regression'
##
##CC.tocsv(y_test_KNN)
#
#
time_end=time.time()
print str(datetime.datetime.now())

time_during=time_end-time_start

print time_during


