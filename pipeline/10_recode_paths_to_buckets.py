import os
import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp

# --------- Parameters --------- #
POOL_SIZE = 39
INPUT_DIR = '/scratch/trj2j/hmm/int/u_paths/'
OUTPUT_DIR = '/scratch/trj2j/hmm/int/u_paths_recode/'

ROLO_S_PATH = './int/rolo_s_clean.pickle'
CLUSTERS_PATH = './int/kmc.sav'
TOP_SITES_PATH = './data/top_sites.csv'
TARGET_SITES_PATH = './data/news.csv'
#PATHS_LIST = './int/cleanup_list.csv'  # fallback if all_paths.csv isn't being used; this needs to just be users

# --------- Load Mapping Info --------- #
print("Loading site and cluster data...")
rolo_s_clean = pickle.load(open(ROLO_S_PATH, 'rb'))
clusters = pickle.load(open(CLUSTERS_PATH, 'rb'))
buckets = clusters.labels_

target_list = pd.read_csv(TARGET_SITES_PATH)['domain'].tolist()
top_list = pd.read_csv(TOP_SITES_PATH)['domain'].tolist()
exclusion_set = set(target_list + top_list)

rolo_s_index = {s: i for i, s in enumerate(rolo_s_clean)}

# --------- Worker Function --------- #
def process_user_path(filename):
    try:
        print(f"🔄 {filename}", flush=True)
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename.replace('.csv', '.parquet'))

        df = pd.read_csv(in_path)
        df.sort_values(by='time', ascending=True, inplace=True)

        # Unique observed sites
        site_values = df[['domain', 'from', 'to']].values.ravel()
        sites_observed = set(pd.unique(site_values)) - exclusion_set
        sites_observed = [s for s in sites_observed if s in rolo_s_index]

        # Remap to buckets
        replacements = {site: str(buckets[rolo_s_index[site]]) for site in sites_observed}
        df.replace(replacements, inplace=True)

        df.to_parquet(out_path, compression=None)
        return f"✅ {filename}"

    except Exception as e:
        return f"❌ {filename}: {e}"

# --------- Main Execution --------- #
if __name__ == '__main__':
    print("Loading file list...")
    #user_paths = pd.read_csv(PATHS_LIST)['paths'].tolist()
    user_paths = os.listdir(INPUT_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing {len(user_paths)} files with {POOL_SIZE} workers...")
    with mp.Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_user_path, user_paths):
            print(result, flush=True)
