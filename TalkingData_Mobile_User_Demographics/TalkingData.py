# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 19:30:51 2016

@author: meil
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

# Any results you write to the current directory are saved as output.

# loading data
events = pd.read_csv("events.csv", dtype = {"device_id": np.str}, infer_datetime_format = True, parse_dates = ["timestamp"])
app_events = pd.read_csv("app_events.csv", usecols = [0, 2, 3],
                            dtype = {"is_active": np.float16, "is_installed": np.float16})

# get hour and drop timestamp
events["hour"] = events["timestamp"].apply(lambda x: x.hour).astype(np.int8)
events.drop("timestamp", axis = 1, inplace = True)

# merge data w/o train or test
events = events.merge(app_events, how = "left", on = "event_id")
#del app_events
events.drop("event_id", axis = 1, inplace = True)

# prep brands
phone = pd.read_csv("phone_brand_device_model.csv", dtype={"device_id": np.str})

# feature hasher
feat = FeatureHasher(n_features=12, input_type="string", dtype=np.float32)
#print(feat)

feat1 = feat.transform(phone["phone_brand"] + " " + phone["device_model"])

#print(feat1) 

events = events.merge(pd.concat([phone["device_id"], pd.DataFrame(feat1.toarray())], axis = 1), how = "left", on = "device_id")

print(events.head(5))

#del phone, feat, feat1

print("pre-merging and hashing finished.")

# train steps
train = pd.read_csv("gender_age_train.csv", dtype = {"device_id": np.str},\
                    usecols = [0, 3])
t2 = train.copy()
train.drop("group", axis = 1, inplace = True)
train = train.merge(events, how = "left", on = "device_id")
train.fillna(-1, inplace = True)
train = train.groupby("device_id").mean().reset_index()
train = train.merge(t2, how ="left", on = "device_id")

le=LabelEncoder()

label = train["group"].copy()
clas=le.fit_transform(label)

train.drop(["group", "device_id"], axis = 1, inplace = True)
del t2

print("train data merged and prepared")
print("-----------------------------------")
print(train.info())
print(train.head(5))
print("-----------------------------------")

# load test data merge with events
test = pd.read_csv("gender_age_test.csv", dtype = {"device_id": np.str})

test = test.merge(events, how = "left", on = "device_id")
#del events
print("test loaded and merged")

test.fillna(-1, inplace = True)
test["hour"] = test["hour"].astype(np.float16)
test = test.groupby("device_id").mean().reset_index()
ids = test["device_id"].copy()
test.drop("device_id", axis = 1, inplace = True)

print("test prepared")
print("-----------------------------------")
print(test.info())
print("-----------------------------------")

#Train

rfc=RandomForestClassifier(n_estimators=10, criterion='gini')
X=train.values
y=clas
rfc.fit(X,y)

y_pred=rfc.predict_proba(test)


sub =pd.concat([ids,pd.DataFrame(y_pred)],axis=1)
#del pred
print("shape of submission: " + str(sub.shape))
sub.to_csv("sample_submission.csv", index = False)
#del sub
print("submission saving finished.")
