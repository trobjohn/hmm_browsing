import numpy as np
import pickle 
import polars as pl
import pandas as pd
import os

os.environ['POLARS_MAX_THREADS'] = '3'

gdf = pd.read_csv('./data/partition.csv', low_memory=False)

for assignment in gdf['assignment']:
    already_done = os.listdir('/scratch/trj2j/hmm/u_paths_partitioned/')
    if '/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'.parquet' not in already_done:
        ## Get list of users:
        with open('/scratch/trj2j/hmm/site_allocations/assignment_'+str(assignment)+'.pickle', 'rb') as file:
            user_list = pickle.load(file)
        user_list = user_list.to_list()

        ## Make a list of lists of user behavior:
        # This is blisteringly fast: Maybe vectorize the lists and pass to polars as dict?
        data = [] 
        for i in range(len(user_list)):
            print(user_list[i])
            try:
                df_i = pd.read_parquet('/scratch/trj2j/hmm/u_paths_recode_01/path_'+str(user_list[i])+'.parquet')
                if df_i.shape[0] > 30:       
                    df_i['user'] = str(user_list[i])
                    data.append(df_i)
            except KeyError:
                print('KeyError')
            except FileNotFoundError:
                print('FileNotFoundError')    

        ## Convert list of lists into a polars df, save as parquet:
        rdf = pl.DataFrame()
        check = 500
        count = 0
        shard = 0
        for k in range(len(data)):
            temp_df = pl.from_pandas(data[k])
            if temp_df.shape[0] > 50 :
                ## These vectors come in as f64 instead of string, if only nan's are detected.abs
                ## Obvious attempts to resolve error result in errors in Polars.
                # temp_df['domain']=temp_df.replace_column(1, temp_df.select(pl.col("domain").cast(pl.String)))
                # temp_df['from']=temp_df.replace_column(2, temp_df.select(pl.col("from").cast(pl.String)))
                # temp_df['to']=temp_df.replace_column(3, temp_df.select(pl.col("to").cast(pl.String)))
                try:
                    rdf = pl.concat([rdf,temp_df]) # XXXX This needs to be optimized somehow.
                    if count == check:
                        est_size = rdf.estimated_size(unit='mb')
                        if est_size > 1024: ## Save and reset
                            rdf.write_parquet('/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'_'+str(shard)+'.parquet')
                            shard += 1
                            count = 0
                            rdf = pl.DataFrame()
                except SchemaError:
                    print('I hate my life.')
            count += 1

        if rdf.shape[0] > 50:
            rdf.write_parquet('/scratch/trj2j/hmm/u_paths_partitioned/grp_'+str(assignment)+'_'+str(shard)+'.parquet')