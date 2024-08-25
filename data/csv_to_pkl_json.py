import pickle
import pandas as pd
import json
import numpy as np


"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#sys.path.append("..")
#import utils
import pickle
import copy
import csv
from datetime import datetime
import time
from io import StringIO
from tqdm import tqdm as tqdm
import pandas as pd


LAT_MIN = 25.0
LAT_MAX = 35.0
LON_MIN = -95.0
LON_MAX = -85.0

#===============

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv_files')
pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pkl_files')
os.makedirs(pkl_path, exist_ok=True)
print(csv_path)
# csv_path = os.path.join(os.getcwd(), 'data/csv_files')
l_csv_filename = [file for file in os.listdir(csv_path) if file.endswith('.csv')]


# Reduce csv file data by removing vessels other than cargo tankers
for i in l_csv_filename:
    print("Current File: ", i)
    input_path = os.path.join(csv_path,i)
    df = pd.read_csv(input_path)
    df = df[(df['VesselType'] >= 70) & (df['VesselType'] <= 90)]
    output_path = os.path.join(csv_path, "ct_" + i )
    df.to_csv(output_path, index=False)

l_csv_filename = [file for file in os.listdir(csv_path) if file.startswith('ct_')] # change the names to relevant (smaller) csv files

# Pickle file

pkl_filename = "ct_dataset_d1_track.pkl"
pkl_filename_train = "ct_dataset_d1_train_track.pkl"
pkl_filename_valid = "ct_dataset_d1_valid_track.pkl"
pkl_filename_test  = "ct_dataset_d1_test_track.pkl"

cargo_tanker_filename = "dataset_d1_cargo_tanker.npy"



LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # the SOG is truncated to 30.0 knots max.

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI, SHIPTYPE = list(range(10))
# MMSI, TIMESTAMP, LAT, LON, SOG, COG, HEADING, Q, W, E, SHIPTYPE, D2C  = list(range(11))
    
print(pkl_filename_train)


## LOADING CSV FILES
#======================================
l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)
n_error = 0
for csv_filename in l_csv_filename:
    data_path = os.path.join(csv_path,csv_filename)
    with open(data_path,"r") as f:
        print("Reading ", csv_filename, "...")
        csvReader = csv.reader(f)
        next(csvReader) # skip the legend row
        count = 1
        for row in csvReader:
            # utc_time = datetime.strptime(row[8], "%Y/%m/%d %H:%M:%S")
            # timestamp = (utc_time - EPOCH).total_seconds()
            utc_time = datetime.fromisoformat(row[1])
            timestamp = (utc_time - EPOCH).total_seconds()
            print(count)
            # print(row)
            count += 1
            # print([int(float(row[0])), int(float(row[10]))])
            try:
                l_l_msg.append([float(row[2]),float(row[3]),
                               float(row[4]),float(row[5]),
                               int(float(row[6])),float(0),
                               int(float(row[11])),int(timestamp),
                               int(float(row[0])),
                               int(float(row[10]))
                               ])
            except:
                n_error += 1
                continue



m_msg = np.array(l_l_msg)
#del l_l_msg
print("Total number of AIS messages: ",m_msg.shape[0])
print("Errors: ", n_error)

print("Lat min: ",np.min(m_msg[:,LAT]), "Lat max: ",np.max(m_msg[:,LAT]))
print("Lon min: ",np.min(m_msg[:,LON]), "Lon max: ",np.max(m_msg[:,LON]))
print("Ts min: ",np.min(m_msg[:,TIMESTAMP]), "Ts max: ",np.max(m_msg[:,TIMESTAMP]))

if m_msg[0,TIMESTAMP] > 1584720228: 
    m_msg[:,TIMESTAMP] = m_msg[:,TIMESTAMP]/1000 # Convert to suitable timestamp format

print("Time min: ",datetime.fromtimestamp(np.min(m_msg[:,TIMESTAMP])))
print("Time max: ",datetime.fromtimestamp(np.max(m_msg[:,TIMESTAMP])))

#timestamps to divide data in train, test, valid
t_min = np.min(m_msg[:,TIMESTAMP])
t_max = np.max(m_msg[:,TIMESTAMP])
t_train_min = t_min
t_train_max = int(t_min + (t_max - t_min) * 0.8) - 1
t_valid_min = t_train_max + 1
t_valid_max = int(t_valid_min + (t_max - t_min) * 0.1) - 1
t_test_min = t_valid_max + 1
t_test_max = t_max

VesselTypes = dict()
l_mmsi = []
n_error = 0
for v_msg in tqdm(m_msg):
    try:
        mmsi_ = v_msg[MMSI]
        type_ = v_msg[SHIPTYPE]
        mmsi_ = int(mmsi_)
        type_ = int(type_)
        if mmsi_ not in l_mmsi :
            VesselTypes[mmsi_] = [type_]
            l_mmsi.append(mmsi_)
        elif type_ not in VesselTypes[mmsi_]:
            VesselTypes[mmsi_].append(type_)
    except:
        n_error += 1
        continue
print("Errors: " , n_error)
for mmsi_ in tqdm(list(VesselTypes.keys())):
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])
    
l_cargo_tanker = list(VesselTypes.keys())

print("Total number of cargos/tankers: ",len(l_cargo_tanker))

print("Saving vessels' type list to ", cargo_tanker_filename)
np.save(cargo_tanker_filename,l_cargo_tanker)


## FILTERING 
#======================================
# Selecting AIS messages in the ROI and in the period of interest.

print("Total msgs: ",len(m_msg))
## LAT LON
m_msg = m_msg[m_msg[:,LAT]>=LAT_MIN]
m_msg = m_msg[m_msg[:,LAT]<=LAT_MAX]
print("Total msgs: ",len(m_msg))
m_msg = m_msg[m_msg[:,LON]>=LON_MIN]
m_msg = m_msg[m_msg[:,LON]<=LON_MAX]
# SOG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,SOG]<=SOG_MAX]
print("Total msgs: ",len(m_msg))
# COG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,COG]<=360]
print("Total msgs: ",len(m_msg))

# TIME
m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]
print("Total msgs: ",len(m_msg))

m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]
print("Total msgs: ",len(m_msg))
m_msg_train = m_msg[m_msg[:,TIMESTAMP]>=t_train_min]
m_msg_train = m_msg_train[m_msg_train[:,TIMESTAMP]<=t_train_max]
m_msg_valid = m_msg[m_msg[:,TIMESTAMP]>=t_valid_min]
m_msg_valid = m_msg_valid[m_msg_valid[:,TIMESTAMP]<=t_valid_max]
m_msg_test  = m_msg[m_msg[:,TIMESTAMP]>=t_test_min]
m_msg_test  = m_msg_test[m_msg_test[:,TIMESTAMP]<=t_test_max]

print("Total msgs: ",len(m_msg))
print("Number of msgs in the training set: ",len(m_msg_train))
print("Number of msgs in the validation set: ",len(m_msg_valid))
print("Number of msgs in the test set: ",len(m_msg_test))


## MERGING INTO DICT
#======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")

# Training set
Vs_train = dict()
for v_msg in tqdm(m_msg_train):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0,9))
    Vs_train[mmsi] = np.concatenate((Vs_train[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_train.keys())):
        Vs_train[key] = np.array(sorted(Vs_train[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Validation set
Vs_valid = dict()
for v_msg in tqdm(m_msg_valid):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_valid.keys())):
        Vs_valid[mmsi] = np.empty((0,9))
    Vs_valid[mmsi] = np.concatenate((Vs_valid[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_valid.keys())):
    Vs_valid[key] = np.array(sorted(Vs_valid[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Test set
Vs_test = dict()
for v_msg in tqdm(m_msg_test):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_test.keys())):
        Vs_test[mmsi] = np.empty((0,9))
    Vs_test[mmsi] = np.concatenate((Vs_test[mmsi], np.expand_dims(v_msg[:9],0)), axis = 0)
for key in tqdm(list(Vs_test.keys())):
    Vs_test[key] = np.array(sorted(Vs_test[key], key=lambda m_entry: m_entry[TIMESTAMP]))




## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
                              [Vs_train,Vs_valid,Vs_test]
                             ):
    print("Writing to ", os.path.join(pkl_path,filename),"...")
    with open(os.path.join(pkl_path,filename),"wb") as f:
        pickle.dump(filedict,f)
    print("Total number of tracks: ", len(filedict))


##JSONIFY
#======================================
def pkl_to_json(path_pkl, path_json):
    with open(path_pkl, 'rb') as file:
        data = pickle.load(file)
    keys_dict = list(data.keys())

    json_data = []
    for key in data.keys():
        # lat = []
        # lon = []
        coords = []
        time = []
        for i in data[key]:
            # lat.append(i[0])
            # lon.append(i[1])
            coords.append([i[0],i[1]])
            time.append(i[7])
        #0,lat,1,lon,7,time
        temp_dict = {
                        "MMSI" : key,
                        # "LAT"  : lat,
                        # "LON"  : lat,
                        "COORDS" : coords,
                        "TIME" : time,
                    }
        json_data.append(temp_dict)

    with open(path_json, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

path_1 = os.path.join(pkl_path, pkl_filename_train)
path_2 = path_1[:-4]+".json"
pkl_to_json(path_1,path_2)
print("JSON saved at: ", path_2)