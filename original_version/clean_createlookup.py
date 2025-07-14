import os as os
import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
import io
import cProfile

rolo_s_clean = pickle.load(open('./models/rolo_s_clean.sav','rb'))
clusters = pickle.load(open('./models/kmc.sav','rb'))
buckets = clusters.labels_

user_paths = pd.read_csv('./all_paths.csv')['paths'].tolist()

