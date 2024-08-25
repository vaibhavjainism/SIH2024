# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Input pipelines script for Tensorflow graph.
This script is adapted from the original script of FIVO.
"""


import numpy as np
import pickle
import os
import sys
# sys.path.append("./data/")
import tensorflow as tf

LAT_BINS = 1000; LON_BINS = 1000; SOG_BINS = 30; COG_BINS = 72
#LAT_BINS = 350; LON_BINS = 1050; SOG_BINS = 30; COG_BINS = 72

def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis):
    def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
        lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
        data_dim = lat_bins + lon_bins + sog_bins + cog_bins
        dense_vect = np.zeros(data_dim)
        dense_vect[int(lat*lat_bins)] = 1.0
        dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
        dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
        dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
        return dense_vect
    dense_msgs = []
    for msg in msgs_:
        dense_msgs.append(create_dense_vect(msg,
                                            lat_bins = LAT_BINS,
                                            lon_bins = LON_BINS,
                                            sog_bins = SOG_BINS ,
                                            cog_bins = COG_BINS))
    dense_msgs = np.array(dense_msgs)
    return dense_msgs, num_timesteps, mmsis


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))


def data_to_mean(data_path):
    try:
        with tf.io.gfile.GFile(data_path, "rb") as f:
            Vs = pickle.load(f)
    except:
        with tf.io.gfile.GFile(data_path, "rb") as f:
            Vs = pickle.load(f, encoding = "latin1")

    data_dim = LAT_BINS + LON_BINS + SOG_BINS + COG_BINS

    mean_path = data_path[:-4] + "_mean.pkl"
    mean_all = np.zeros((data_dim,))
    sum_all = np.zeros((data_dim,))
    total_ais_msg = 0

    current_mean = np.zeros((0,data_dim))
    current_ais_msg = 0

    count = 0
    for mmsi in list(Vs.keys()):
        count += 1
        print(count)
        tmp = Vs[mmsi][:,[LAT,LON,SOG,COG]]
        tmp[tmp == 1] = 0.99999
        current_sparse_matrix,_,_ = sparse_AIS_to_dense(tmp,0,0)
    #    current_mean = np.mean(current_sparse_matrix,axis = 0)
        sum_all += np.sum(current_sparse_matrix,axis = 0)
        total_ais_msg += len(current_sparse_matrix)

    mean = sum_all/total_ais_msg
    print(mean_path)
    with open(mean_path,"wb") as f:
        pickle.dump(mean,f)


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
l_filename = [file for file in os.listdir(dataset_path) if file.endswith(".pkl")]
print(dataset_path)
for i in range(len(l_filename)):
    l_filename[i] = os.path.join(dataset_path,l_filename[i])
    print(l_filename[i])

for i in l_filename:
    data_to_mean(i)

