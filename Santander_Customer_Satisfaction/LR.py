# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:32:54 2016

@author: meil

Santander Customer Satisfaction

"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble

import time

class SCS(object):
    
    def __init__(self):
        self.trainset=[]
        self.trainlabel=[]
        self.trainset_new=[]
        self.testset=[]
        self.testset_new=[]
        self.testlabel=[]
   
    def loaddataset(self,path,mode):
       data=pd.read_csv(path)
       dataset=[]
       dataset=data.values
       
       if mode=='train':
           self.trainlabel=dataset[0:,-1]
           self.trainset=dataset[0:,1:-1]
       elif mode=='test':
           self.testset=dataset[0:,1:]
        
    def decompose(self,trainset,k):
        pca=PCA(n_components=k)
        pca.fit(trainset)
        self.trainset_new=pca.transform(trainset)
        return pca
        
    def trainmodel_LR(self):
        classifier_LR=LogisticRegression()
        
        samples=self.trainset_new
        target=self.trainlabel
        
        classifier_LR.fit(samples,target)
        return classifier_LR
    
    def trainmodel_SVM(self):
        classifier_SVM=svm.SVC(kernel='rbf')
        samples=self.trainset_new
        target=self.trainlabel
        classifier_SVM.fit(samples,target)
        return classifier_SVM
    
    def trainmodel_KNN(self):
        classifier_KNN=KNeighborsClassifier(n_neighbors=4)
        samples=self.trainset_new
        target=self.trainlabel
        classifier_KNN.fit(samples,target)
        return classifier_KNN
    
    def trainmodel_RF(self):
        classifier_RF=ensemble.RandomForestClassifier(n_estimators=20)
        samples=self.trainset_new
        target=self.trainlabel
        classifier_RF.fit(samples,target)
        
        return classifier_RF
    
    def evaluate_overfiting(self,labels):
        m=np.shape(labels)[0]
        count=0
        for i in xrange(m):
            if self.trainlabel[i]==labels[i]:
                count=count+1
        return count/float(m) 
        
    def evaluate_test(self,labels_LR,labels_SVM):
        m=np.shape(labels_LR)[0]
        count=0
        for i in xrange(m):
            if labels_SVM[i]==labels_LR[i]:
                count=count+1
        
        return count/float(m)
        
    def cross_validation(self,k,method):
        
        dataset=self.trainset_new
        labels=self.trainlabel
        kf = KFold(np.shape(dataset)[0], n_folds=k)
        
        precision=np.zeros((k,1))
        
        i=0
        
        for train_index,test_index in kf:
            X_train, X_test = dataset[train_index], dataset[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            if method=='KNN':
                sub_classifier_KNN=KNeighborsClassifier(n_neighbors=3)
                sub_classifier_KNN.fit(X_train,y_train)
                y_test_predict=sub_classifier_KNN.predict(X_test)
                precision[i]=self.evaluate_cross_validation(y_test,y_test_predict)
            elif method=="LR":
                sub_classifier_LR=LogisticRegression()
                sub_classifier_LR.fit(X_train,y_train)
                y_test_predict=sub_classifier_LR.predict(X_test)
                precision[i]=self.evaluate_cross_validation(y_test,y_test_predict)
            elif method=="SVM":
                sub_classifier_SVM=svm.SVC(kernel='rbf')
                sub_classifier_SVM.fit(X_train,y_train)
                y_test_predict=sub_classifier_SVM.predict(X_test)
                precision[i]=self.evaluate_cross_validation(y_test,y_test_predict)
                
                i=i+1
            
        return precision     
        
    def evaluate_cross_validation(self,labels_actual,labels_predict):
        m=np.shape(labels_actual)[0]
        count=0
        for i in xrange(m):
            if labels_predict[i]==labels_actual[i]:
                count=count+1
        return float(count)/m
        
    def under_sampling(self):
        dataset=self.trainset_new
        label=self.trainlabel
        count0=label.tolist().count(0)
        count1=label.tolist().count(1)
        m=np.shape(dataset)[0]
        trainset1=[dataset[i] for i in xrange(m) if label[i]==1]
        trainset0=[dataset[i] for i in xrange(m) if label[i]==0]
        
        trainset=np.concatenate((trainset0[:count1],trainset1))
        trainlabel=np.concatenate((np.zeros((count1,1)),np.ones((count1,1))))
        
        return trainset,trainlabel
     
    def trainmodel_KNN_undersampling(self,trainset,trainlabel):
        samples=trainset
        target=trainlabel
        classifier_KNN_undersampling=KNeighborsClassifier(n_neighbors=3)
        classifier_KNN_undersampling.fit(samples,target)
        
        return classifier_KNN_undersampling
    
    def trainmodel_LR_undersampling(self,trainset,trainlabel):
        samples=trainset
        target=trainlabel
        classifier_LR_undersampling=LogisticRegression()
        classifier_LR_undersampling.fit(samples,target)
        
        return classifier_LR_undersampling
    
    def trainmodel_SVM_undersampling(self,trainset,trainlabel):
        samples=trainset
        target=trainlabel
        classifier_SVM_undersampling=svm.SVC(kernel='rbf')
        classifier_SVM_undersampling.fit(samples,target)
        
        return classifier_SVM_undersampling
        
    def normalization_L2(self):
       m0=np.shape(self.trainset)[0]
       m1=np.shape(self.testset)[0]
       dataset=np.concatenate((self.trainset,self.testset))
       dataset_normalized=normalize(dataset, norm='l2',axis=0)
       
       X_normalized_0=dataset_normalized[:m0]
       X_normalized_1=dataset_normalized[:m1]
       
       return X_normalized_0,X_normalized_1
      
    def scaling(self,method):
                
       m0=np.shape(self.trainset)[0]
       m1=np.shape(self.testset)[0]
       dataset=np.concatenate((self.trainset,self.testset))
              
       if method=='scale':
            dataset_scaled=scale(dataset)
            
       elif method=='minmax':
            min_max_scaler=MinMaxScaler()
            dataset_scaled=min_max_scaler.fit_transform(dataset)
            
       X_scaled_0=dataset_scaled[:m0]
       X_scaled_1=dataset_scaled[:m1]
       
       return X_scaled_0,X_scaled_1
        
#==============================================================================
# Class
#==============================================================================
      
time_start=time.time()

scs=SCS()

#<-------------trainset----------------------->
path="train.csv"

scs.loaddataset(path,'train')

trainset=scs.trainset

pca=scs.decompose(trainset,100)

trainset=scs.trainset_new

#<-------------testset----------------------->

scs.loaddataset("test.csv",'test')

testset=scs.testset

testset=pca.transform(testset)

inX=testset

#<-------------LR----------------------->
#classifier_LR=scs.trainmodel_LR()
#
#labels_LR=classifier_LR.predict(inX)
#
#labels_train_LR=classifier_LR.predict(trainset)
#
#overfiting_LR=scs.evaluate_overfiting(labels_train_LR)

#<-------------SVM----------------------->
#classifier_SVM=scs.trainmodel_SVM()
#
#labels_SVM=classifier_SVM.predict(inX)
#
#labels_train_SVM=classifier_SVM.predict(trainset)
#
#overfiting_SVM=scs.evaluate_overfiting(labels_train_SVM)

#match_LR_SVM=scs.evaluate_test(labels_LR,labels_SVM)

#<-------------KNN---------------------->

#classifier_KNN=scs.trainmodel_KNN()
#
#labels_KNN=classifier_KNN.predict(inX)
#
#labels_train_KNN=classifier_KNN.predict(trainset)

#overfitting_KNN=scs.evaluate_overfiting(labels_train_KNN)


#<-------------Random Forest---------------------->

#classifier_RF=scs.trainmodel_RF()
#
#labels_RF=classifier_RF.predict(inX)
#
#labels_train_RF=classifier_RF.predict(trainset)
#
#print classifier_RF.score(trainset,scs.trainlabel)

#<-------------Time-----------      >

#time_end=time.time()
#
#time_during=time_end-time_start
#
#print time_during

#<-------------GradientBoosting---------------------->

#classifier_GB=ensemble.GradientBoostingClassifier(n_estimators=1000, max_leaf_nodes=4, max_depth= None, random_state= 2,min_samples_split= 5)
#target=scs.trainlabel
#classifier_GB.fit(trainset,target)
#
#lables_test_GB=classifier_GB.predict(inX)


#<--------------Cross-Validation in trainset------------------------------->

#precision_KNN=scs.cross_validation(25,'KNN')
#precision_SVM=scs.cross_validation(7,'SVM')
#precision_LR=scs.cross_validation(7,'LR')

#print precision_KNN
#print precision_SVM
#print precision_LR

#<--------------imbalance in trainset------------------------------->

#under-sampling

#trainset,trainlabel=scs.under_sampling()
#
#classifier_LR_undersampling=scs.trainmodel_LR_undersampling(trainset,trainlabel)
#
#labels_LR_undersampling=classifier_LR_undersampling.predict(inX)

#<--------------normalization------------------------------->
#MinMax-Scaling

#trainset_minmax,testset_minmax=scs.scaling('scale')
#
#
#pca=PCA(n_components=100)
#
#pca.fit(trainset_minmax)
#
#trainset_minmax_new=pca.transform(trainset_minmax)
#testset_minmax_new=pca.transform(testset_minmax)
#
#
#classifier_LR=LogisticRegression()
#
#samples=trainset_minmax_new
#
#target=scs.trainlabel
#
#classifier_LR.fit(samples,target)
#
#labels_test_minmax=classifier_LR.predict(testset_minmax_new)




#<--------------overall------------------------------->

trainset_minmax,testset_minmax=scs.scaling('minmax')

pca=PCA(n_components=100)
pca.fit(trainset_minmax)
trainset_minmax_new=pca.transform(trainset_minmax)
testset_minmax_new=pca.transform(testset_minmax)

dataset=trainset_minmax_new
label=scs.trainlabel
count0=label.tolist().count(0)
count1=label.tolist().count(1)
m=np.shape(dataset)[0]
trainset1=[dataset[i] for i in xrange(m) if label[i]==1]
trainset0=[dataset[i] for i in xrange(m) if label[i]==0]
        
trainset=np.concatenate((trainset0[:count1],trainset1))
trainlabel=np.concatenate((np.zeros((count1,1)),np.ones((count1,1))))

samples=trainset
target=trainlabel

classifier_GB=ensemble.GradientBoostingClassifier(n_estimators=1000, max_leaf_nodes=4, max_depth= None, random_state= 2,min_samples_split= 5)

classifier_GB.fit(samples,target)

lables_test_GB=classifier_GB.predict_proba(inX)




