import pandas as pd 
import os as os
import numpy as np
import seaborn as sns
import multiprocessing as mp
import pickle

def process_partition_element(index):
    #files = os.listdir('/scratch/trj2j/hmm/u_paths_partitioned/')
    ## Construct page space
    # rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
    # adf = pd.read_csv('./sweep/kmc_clusters.csv') # idio sites to clusters
    # top = pd.read_csv('./data/top_sites.csv') # idio sites to clusters
    # target = pd.read_csv('./data/news.csv')
    # target_sites = target['domain']
    # clusters = adf['cluster'].unique()
    # top_sites = top['domain']
    # sites = [None] + target_sites.to_list() \
    #     + top_sites.to_list() \
    #     + ( clusters.astype('str') ).tolist()
    # #
    # with open('/scratch/trj2j/hmm/todo.pkl', 'rb') as file:
    #     todo_files = pickle.load(file)
    # element_path = todo_files[index]
    # #
    # this_df = pd.read_parquet('/scratch/trj2j/hmm/u_paths_partitioned/'+element_path)
    # this_df.sort_values(['user','time'],axis=0)
    # session_window = 9 # Minutes
    # this_df['delta'] = this_df.loc[:,['user','time'] ].groupby('user').diff()/60 #.diff('time')
    # this_df['session_start'] = this_df['delta']>session_window
    # this_df['session'] = this_df.loc[:,['user','session_start'] ].groupby('user').cumsum()
    # sites_unique = list(set(this_df['to']).union(set(this_df['domain'])))
    # index_corr = [ sites.index( sites_unique[i] ) for i in range(len(sites_unique)) ]
    # this_df['do_ind'] = np.zeros( len(this_df['domain']) )
    # this_df['to_ind'] = np.zeros( len(this_df['to']) )
    # for k in range(len(sites_unique)):
    #     this_df.loc[ this_df['domain'] == sites_unique[k], 'do_ind' ] = index_corr[k]
    #     this_df.loc[ this_df['to'] == sites_unique[k], 'to_ind' ] = index_corr[k]
    # this_df.loc[:,['user','domain','do_ind','to','to_ind','session'] ].to_parquet('/scratch/trj2j/hmm/u_paths_hmm/'+element_path)

    rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
    adf = pd.read_csv('./sweep/kmc_clusters.csv') # idio sites to clusters
    top = pd.read_csv('./data/top_sites.csv') # idio sites to clusters
    target = pd.read_csv('./data/news.csv')
    target_sites = target['domain']
    clusters = adf['cluster'].unique()
    top_sites = top['domain']
    sites = [None] + target_sites.to_list() \
        + top_sites.to_list() \
        + ( clusters.astype('str') ).tolist()

    with open('/scratch/trj2j/hmm/todo.pkl', 'rb') as file:
        todo_files = pickle.load(file)
    element_path = todo_files[index]
    #
    this_df = pd.read_parquet('/scratch/trj2j/hmm/u_paths_partitioned/'+element_path)

    sites_map = pd.Series(np.arange(len(sites)), index=sites)
    this_df['fr_x'] = this_df['from'].map(sites_map).fillna(-1).astype(int)
    this_df['do_x'] = this_df['domain'].map(sites_map).fillna(-1).astype(int)
    this_df['to_x'] = this_df['to'].map(sites_map).fillna(-1).astype(int)

    session_window = 9 # Minutes   
    this_df['delta'] = this_df.loc[:,['user','time'] ].groupby('user').diff()/60 #.diff('time')
    this_df['session_start'] = this_df['delta']>session_window
    this_df['session'] = this_df.loc[:,['user','session_start'] ].groupby('user').cumsum()

    this_df.to_parquet('/scratch/trj2j/hmm/u_paths_hmm/'+element_path)

    return element_path


def main():
    print('Loading objects and data')

    POOL_SIZE = 30  # Rivanna cores

    ## Directory of u_paths_partitioned files
    all_files = os.listdir('/scratch/trj2j/hmm/u_paths_partitioned/')
    done_files = os.listdir('/scratch/trj2j/hmm/u_paths_hmm/')
    todo_files = list(set(all_files)-set(done_files))
    pickle.dump(todo_files, open('/scratch/trj2j/hmm/todo.pkl', 'wb')) 
 

    print('Running code')
    todo_files = todo_files[:35] # Avoid deadlocking
    with mp.Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_partition_element, range(len(todo_files))):
            print('Completed: ', result, flush=True)

if __name__ == '__main__':
    main()



