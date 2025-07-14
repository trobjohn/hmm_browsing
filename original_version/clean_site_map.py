# %%

import pandas as pd
import os
import numpy as np
import multiprocessing as mp
import traceback
import pickle

# --------- Shared Preload (Pandas-friendly) --------- #
rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
adf = pd.read_csv('./sweep/kmc_clusters.csv')
top = pd.read_csv('./data/top_sites.csv')
target = pd.read_csv('./data/news.csv')

target_sites = target['domain']
clusters = adf['cluster'].unique()
top_sites = top['domain']

sites = [None] + target_sites.to_list() + top_sites.to_list() + clusters.astype(str).tolist()
sites_map = pd.Series(np.arange(len(sites)), index=sites)


# %%

sites_map.to_csv('./data/sites_map.csv')

sites_map.to_csv('/scratch/trj2j/hmm/sites_map.csv')


# %%
