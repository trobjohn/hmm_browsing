import numpy as np
import pickle 
import polars as pl
import pandas as pd
import os
import sys

## 25GB RAM

os.environ['POLARS_MAX_THREADS'] = '3'

gdf = pd.read_csv('./data/partition.csv')

assignment = gdf['assignment'][0]

for assignment in gdf['assignment']:
    already_done = os.listdir('/scratch/trj2j/hmm/u_paths_partitioned/')
    if '/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'.parquet' not in already_done:
        ## Get list of users:
        with open('/scratch/trj2j/hmm/site_allocations/assignment_'+str(assignment)+'.pickle', 'rb') as file:
            user_list = pickle.load(file)
        user_list = user_list.to_list()

        ## Make a list of lists of user behavior:
        data = []
        check = 2500
        count = 0
        shard = 0
        user_count = 0
        for i in range(len(user_list)):
            print(user_list[i])
            try:
                df_i = pd.read_parquet('/scratch/trj2j/hmm/u_paths_recode_01/path_'+str(user_list[i])+'.parquet')
                if df_i.shape[0] > 35:       
                    df_i['user'] = str(user_list[i])
                    #df_i = df_i.with_columns(pl.lit(str(user_list[i])).alias("user"))
                    #df_i.insert_column(5, pl.lit(str(user_list[i])).alias("user"))
                    data.append(df_i)
                    user_count =+ 1
                    count += 1
                    if count == check:
                        est_size = sys.getsizeof(data)
                        print('Estimated size: ', est_size)
                        count = 0
                        if est_size > 1e9 :
                            rdf = pd.concat(data,axis=0,ignore_index=True)
                            rdf.to_parquet('/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'_'+str(shard)+'.parquet')
                            shard += 1
                            data = []
                            count = 0
                            user_count = 0
            except KeyError:
                print('KeyError')
            except FileNotFoundError:
                print('FileNotFoundError')    

        if user_count > 0:
            rdf = pd.concat(data,axis=0,ignore_index=True)
            rdf.to_parquet('/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'_'+str(shard)+'.parquet')
