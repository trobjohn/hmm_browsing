import pandas as pd 
import os as os
import numpy as np
import seaborn as sns
import multiprocessing as mp
import pickle

def process_partition_element(index):
    ## Construct page space
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

    #files = os.listdir('/scratch/trj2j/hmm/u_paths_partitioned/')
    # with open('/scratch/trj2j/hmm/todo.pkl', 'rb') as file:
    #     todo_files = pickle.load(file)
    with open('/scratch/trj2j/hmm/todo.pkl', 'rb') as file:
        todo_files = pickle.load(file)
    
    ## Group to analyze and page index sequence
    dir = '/scratch/trj2j/hmm/u_paths_hmm/'
    this_file = todo_files[index]
    print('This file: ', this_file, flush=True)
    df = pd.read_parquet(dir+this_file)
    

    #sites_unique = df['domain'].unique().tolist()  # We want a canonical correspondence
    sites_unique = list(set(df['to']).union(set(df['domain'])))
    index_corr = [ sites.index( sites_unique[i] ) for i in range(len(sites_unique)) ]

    df['do_ind'] = np.zeros( len(df['domain']) )
    df['to_ind'] = np.zeros( len(df['to']) )

    for k in range(len(sites_unique)):
        df.loc[ df['domain'] == sites_unique[k], 'do_ind' ] = index_corr[k]
        df.loc[ df['to'] == sites_unique[k], 'to_ind' ] = index_corr[k]

    #
    out_file = '/scratch/trj2j/hmm/u_paths_hmm_xt/'+'grp_'+this_file
    df.to_parquet(out_file)
    return element_path


def main():
    print('Loading objects and data')

    POOL_SIZE = 25  # Rivanna cores

    ## Directory of u_paths_partitioned files
    all_files = os.listdir('/scratch/trj2j/hmm/u_paths_hmm/')
    done_files = os.listdir('/scratch/trj2j/hmm/u_paths_hmm_xt/')
    todo_files = list(set(all_files)-set(done_files))
    pickle.dump(todo_files, open('/scratch/trj2j/hmm/todo.pkl', 'wb')) 
 
    #print(todo_files)

    print('Running code')
    with mp.Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_partition_element, range(len(todo_files))):
            print('Completed: ', result, flush=True)

if __name__ == '__main__':
    main()



