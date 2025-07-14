import os as os
import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
import io

# 1. Get a list of all the sites referenced that aren't nan
# 2. Bulk replace them throughout the df

def process_user_path(u_p):
    print(u_p,flush=True)
    df_u = pd.read_csv(RIVANNA_PATH+u_p)
    df_u = df_u.sort_values(by = 'time',ascending = True)
    sites = pd.unique( df_u.loc[:,['domain','from','to'] ].values.ravel() ) # Sites observed
    sites = list( set(sites).intersection(set(rolo_s_clean)) ) # Drop target sites
    corr = buckets[ [ rolo_s_clean.index(i) for i in sites] ] # Map sites to buckets
    for k in range(len(sites)):
        df_u.replace(sites[k], corr[k].astype('str'), inplace=True)
    df_u.to_parquet(SCRATCH_PATH+u_p.removesuffix('.csv')+'.parquet',compression =None)

if __name__ == '__main__':
    
    print('Loading objects and data')
    POOL_SIZE = 39
    RIVANNA_PATH = '/scratch/trj2j/hmm/u_paths/'
    SCRATCH_PATH = '/scratch/trj2j/hmm/u_paths_recode_01/'
    # Load saved cluster results
    rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb'))
    clusters = pickle.load(open('./sweep/kmc.sav','rb'))
    buckets = clusters.labels_

    target = pd.read_csv('./data/news.csv')
    target_list = target['domain']
    top = pd.read_csv('./data/top_sites.csv')
    top_list = top['domain']
    unique_list = target_list.to_list() + top_list.to_list()

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    #user_paths = pd.read_csv('./all_paths.csv')['paths'].tolist() # Original run
    user_paths = pd.read_csv('./sweep/cleanup_list.csv')['paths'].tolist() # Clean-up runs
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    ## Main recoding
    POOL = mp.Pool()
    POOL.map_async(process_user_path,user_paths)
    POOL.close()
    POOL.join()

    # ## Clean up edge cases
    # result = [process_user_path(u_p) for u_p in user_paths]

# import cProfile
# up0 = user_paths[0]
# cProfile.run('process_user_path(up0)')