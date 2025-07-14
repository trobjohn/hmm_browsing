import os as os
import gzip as gz
import pandas as pd
import multiprocessing as mp
import itertools

def get_sites(file):
    df = pd.read_csv(rivanna_path+'./u_paths/'+file,low_memory=False)
    sites = set(df['domain']).union(set(df['from'])).union(set(df['to'])) # HERE WAS THE BUG
    return(sites)

if __name__ == '__main__':

    POOL_SIZE = 20 # 16-32 locally
    #rivanna_path = './'
    rivanna_path = '/scratch/trj2j/hmm/'
    files = os.listdir(rivanna_path+'u_paths/')
    gdf = pd.DataFrame({'paths':files})
    gdf.to_csv('all_paths.csv',index=False)

    # Generate list of sites:
    print('Creating sitelist...')
    mp.set_start_method("spawn")
    POOL = mp.Pool(POOL_SIZE)
    results = POOL.map(get_sites,files)
    POOL.close()
    POOL.join()

    print('Merging sets to list and saving...')

    merged = set(itertools.chain.from_iterable(results))
    gdf = pd.DataFrame({'sites':list(merged)})
    #gdf = gdf.drop_duplicates('sites')
    gdf.to_csv('./sweep/rolodex_sites.csv')