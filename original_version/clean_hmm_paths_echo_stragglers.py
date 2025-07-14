
import os
import pandas as pd
import numpy as np
import traceback
import pickle

input_dir = '/scratch/trj2j/hmm/u_paths_partitioned/'
output_dir = '/scratch/trj2j/hmm/u_paths_hmm/'

# Just these 4 files
# retry_files = [
#     "grp_165_0.parquet",
#     "grp_30_0.parquet",
#     "grp_366_0.parquet",
#     "grp_390_0.parquet",
# ] # Original problem files

retry_files = ['grp_117_0.parquet',
    'grp_120_0.parquet',
    'grp_141_0.parquet',
    'grp_168_0.parquet',
    'grp_360_0.parquet',
    'grp_369_0.parquet',
    'grp_90_0.parquet',] # Final pass problem files

#with open("remaining_stragglers.txt") as f:
#    retry_files = [line.strip() for line in f]


# Preload shared objects
rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
adf = pd.read_csv('./sweep/kmc_clusters.csv')
top = pd.read_csv('./data/top_sites.csv')
target = pd.read_csv('./data/news.csv')

target_sites = target['domain']
clusters = adf['cluster'].unique()
top_sites = top['domain']
sites = [None] + target_sites.to_list() + top_sites.to_list() + clusters.astype(str).tolist()
sites_map = pd.Series(np.arange(len(sites)), index=sites)

def process_file(filename):
    try:
        final_path = os.path.join(output_dir, filename)
        tmp_path = final_path + ".tmp"

        if os.path.exists(final_path):
            print(f"✔ Skipping {filename} (already done)")
            return

        print(f"🚧 Processing {filename}...")
        df = pd.read_parquet(os.path.join(input_dir, filename))

        df['fr_x'] = df['from'].map(sites_map).fillna(-1).astype(int)
        df['do_x'] = df['domain'].map(sites_map).fillna(-1).astype(int)
        df['to_x'] = df['to'].map(sites_map).fillna(-1).astype(int)

        session_window = 9
        df['delta'] = df[['user', 'time']].groupby('user').diff() / 60
        df['session_start'] = df['delta'] > session_window
        df['session'] = df[['user', 'session_start']].groupby('user').cumsum()

        df.to_parquet(tmp_path)
        os.rename(tmp_path, final_path)
        print(f"✅ Done: {filename}")
    
    except Exception as e:
        print(f"❌ Error on {filename}:\n{traceback.format_exc()}")
        with open('errors_stragglers.log', 'a') as f:
            f.write(f"[{filename}]\n{traceback.format_exc()}\n\n")

if __name__ == "__main__":
    for f in retry_files:
        process_file(f)
