# coding=utf8
"""
Created on Tue Jul 12 11:53:08 2016

@author: meil
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher

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
phone = pd.read_csv("phone_brand_device_model.csv", dtype={"device_id": np.str},usecols = [0, 1, 2])


#feat = FeatureHasher(n_features=12, input_type="string")
#
#feat1 = feat.transform(phone["phone_brand"] + " " + phone["device_model"])

events = events.merge(pd.concat([phone["device_id"], phone["phone_brand"],phone["device_model"]], axis = 1), how = "left", on = "device_id")
#del phone, feat, feat1

print("pre-merging and hashing finished.")

# train steps
train = pd.read_csv("E:\MachineLearning\Kaggle\ML.Kaggle\TalkingData_Mobile_User_Demographics\gender_age_train.csv", dtype = {"device_id": np.str},\
                    usecols = [0, 3])
t2 = train.copy()
train.drop("group", axis = 1, inplace = True)
train = train.merge(events, how = "left", on = "device_id")
train.fillna(-1, inplace = True)
train = train.groupby("device_id").mean().reset_index()
train = train.merge(t2, how ="left", on = "device_id")

label = train["group"].copy()
train.drop(["group", "device_id"], axis = 1, inplace = True)
#del t2

print("train data merged and prepared")
print("-----------------------------------")
#td.mergedata('test')

##################
#   App Events
##################
# remove duplicates(app_id)

##################
#     Events
##################




# remove duplicates(app_id)

# expand to multiple rows

##################
#   Phone Brand
##################


##################
#  Train and Test
##################




print(train.info())
print("-----------------------------------")