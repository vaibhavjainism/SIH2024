# In[1]:

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import os
from tqdm import tqdm_notebook as tqdm
sys.path.append("..")
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO

from tqdm import tqdm
import argparse

# In[2]:
def track_to_pkl(dataset_path,input_path,output_path):
    def getConfig(args=sys.argv[1:]):
        parser = argparse.ArgumentParser(description="Parses command.")
        
        # File paths
        parser.add_argument("--dataset_dir", type=str, 
                            default=dataset_path,
                            help="Dir to dataset.")    
        parser.add_argument("--l_input_filepath", type=str, nargs='+',
                            default=[input_path],
                            help="List of path to input files.")
        parser.add_argument("--output_filepath", type=str,
                            default=output_path,
                            help="Path to output file.")
        
        parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
        config = parser.parse_args(args)
        return config

    config = getConfig(sys.argv[1:])
    #=====================================================================
    # LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = config.lat_min,config.lat_max,config.lon_min,config.lon_max
    LAT_MIN = 25.0
    LAT_MAX = 35.0
    LON_MIN = -95.0
    LON_MAX = -85.0
    LAT_RANGE = LAT_MAX - LAT_MIN
    LON_RANGE = LON_MAX - LON_MIN
    SPEED_MAX = 30.0  # knots
    DURATION_MAX = 24 #h

    LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

    FIG_W = 960
    FIG_H = int(960*LAT_RANGE/LON_RANGE) #533 #768

    dict_list = []
    for filename in config.l_input_filepath:
        with open(os.path.join(filename),"rb") as f:
            temp = pickle.load(f)
            dict_list.append(temp)


    # In[3]:
    print(" Remove erroneous speeds...")
    Vs = dict()
    for Vi,filename in zip(dict_list, config.l_input_filepath):
        print(filename)
        for mmsi in list(Vi.keys()):       
            # Boundary
            lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                    (Vi[mmsi][:,LAT] < LAT_MIN))
            Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]
            lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                    (Vi[mmsi][:,LON] < LON_MIN))
            Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]
            abnormal_speed_idx = Vi[mmsi][:,SOG] > SPEED_MAX
            Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_speed_idx)]
            # Deleting empty keys
            if len(Vi[mmsi]) == 0:
                del Vi[mmsi]
                continue
            if mmsi not in list(Vs.keys()):
                Vs[mmsi] = Vi[mmsi]
                del Vi[mmsi]
            else:
                Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
                del Vi[mmsi]
    del dict_list, Vi, abnormal_speed_idx


    # In[4]:
    print(len(Vs))

    # In[5]:


    ## STEP 2: VOYAGES SPLITTING 
    #======================================
    # Cutting discontiguous voyages into contiguous ones
    print("Cutting discontiguous voyages into contiguous ones...")
    count = 0
    voyages = dict()
    INTERVAL_MAX = 2*3600 # 2h
    for mmsi in list(Vs.keys()):
        v = Vs[mmsi]
        # Intervals between successive messages in a track
        intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
        idx = np.where(intervals > INTERVAL_MAX)[0]
        if len(idx) == 0:
            voyages[count] = v
            count += 1
        else:
            tmp = np.split(v,idx+1)
            for t in tmp:
                voyages[count] = t
                count += 1


    # In[6]:


    print(len(Vs))


    # In[7]:


    # STEP 3: REMOVING SHORT VOYAGES
    #======================================
    # Removing AIS track whose length is smaller than 20 or those last less than 4h
    print("Removing AIS track whose length is smaller than 20 or those last less than 4h...")

    for k in list(voyages.keys()):
        duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
        if (len(voyages[k]) < 20) or (duration < 4*3600):
            voyages.pop(k, None)


    # In[8]:


    print(len(voyages))


    # In[13]:


    ## STEP 4: SAMPLING
    #======================================
    # Sampling, resolution = 5 min
    print('Sampling...')
    Vs = dict() 
    count = 0
    for k in tqdm(list(voyages.keys())):
        v = voyages[k]  
        sampling_track = np.empty((0, 9))
        for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
            tmp = utils.interpolate(t,v)
            if tmp is not None:
                sampling_track = np.vstack([sampling_track, tmp])
            else:
                sampling_track = None
                break
        if sampling_track is not None:
            Vs[count] = sampling_track
            count += 1

    # In[11]:


    ## STEP 5: RE-SPLITTING
    #======================================
    print('Re-Splitting...')
    Data = dict()
    count = 0
    for k in tqdm(list(Vs.keys())): 
        v = Vs[k]
        # Split AIS track into small tracks whose duration <= 1 day
        idx = np.arange(0, len(v), 12*DURATION_MAX)[1:]
        tmp = np.split(v,idx)
        for subtrack in tmp:
            # only use tracks whose duration >= 4 hours
            if len(subtrack) >= 12*4:
                Data[count] = subtrack
                count += 1
    print(len(Data))
       
    ## STEP 6: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
    #======================================
    # Removing 'moored' or 'at anchor' voyages
    print("Removing 'moored' or 'at anchor' voyages...")
    for k in  tqdm(list(Data.keys())):
        d_L = float(len(Data[k]))

        if np.count_nonzero(Data[k][:,NAV_STT] == 1)/d_L > 0.7 \
        or np.count_nonzero(Data[k][:,NAV_STT] == 5)/d_L > 0.7:
            Data.pop(k,None)
            continue
        sog_max = np.max(Data[k][:,SOG])
        if sog_max < 1.0:
            Data.pop(k,None)
    print(len(Data))
    # In[15]:


    ## STEP 7: REMOVING LOW SPEED TRACKS
    #======================================
    print("Removing 'low speed' tracks...")
    for k in tqdm(list(Data.keys())):
        d_L = float(len(Data[k]))
        if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > 0.8:
            Data.pop(k,None)
    print(len(Data))

    ## STEP 9: NORMALISATION
    #======================================
    print('Normalisation...')
    for k in tqdm(list(Data.keys())):
        v = Data[k]
        v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
        v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
        v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
        v[:,SOG] = v[:,SOG]/SPEED_MAX
        v[:,COG] = v[:,COG]/360.0


    # In[22]:

    config.output_filepath = os.path.join(config.dataset_dir, config.output_filepath)
    print(config.output_filepath)


    # In[23]:


    # plt.plot(Data[0][:,LON],Data[0][:,LAT])


    # In[24]:


    print(len(Data))

    # In[28]:


    ## STEP 10: WRITING TO DISK
    #======================================
    with open(config.output_filepath,"wb") as f:
        pickle.dump(Data,f)


    # In[29]:


    # print(debug)


    # In[30]:


    print(len(Data))


    # In[31]:


    minlen = 1000
    for k in list(Data.keys()):
        v = Data[k]
        if len(v) < minlen:
            minlen = len(v)
    print("min len: ",minlen)

    # In[36]:


    Vs = Data
    FIG_DPI = 150
    plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
    cmap = plt.get_cmap('Blues')
    l_keys = list(Vs.keys())
    N = len(Vs)
    for d_i in range(N):
        key = l_keys[d_i]
        c = cmap(float(d_i)/(N-1))
        tmp = Vs[key]
        v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
        v_lon = tmp[:,1]*LON_RANGE + LON_MIN
    #     plt.plot(v_lon,v_lat,linewidth=0.8)
        plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

    ## Coastlines
    # if "bretagne" in config.output_filepath:
    #     for point in l_coastline_poly:
    #         poly = np.array(point)
    #         plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(config.output_filepath.replace(".pkl",".png"))




# tracks to preprocessed data
pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pkl_files')
l_track_filename = [file for file in os.listdir(pkl_path) if file.endswith(".pkl")]
l_output_filename = []
print(pkl_path)
for i in range(len(l_track_filename)):
    temp = l_track_filename[i].split("_track")
    l_output_filename.append(temp[0]+temp[1])
    l_track_filename[i] = os.path.join(pkl_path,l_track_filename[i])
    print(l_track_filename[i])
print(l_output_filename)
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(dataset_path, exist_ok=True)

for i in range(len(l_track_filename)):
    track_to_pkl(dataset_path, l_track_filename[i], l_output_filename[i])

