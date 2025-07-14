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

input_dir = '/scratch/trj2j/hmm/u_paths_partitioned/'
output_dir = '/scratch/trj2j/hmm/u_paths_hmm/'

# --------- Hardened Worker Function --------- #
def process_partition_element(filename):
    try:
        final_path = os.path.join(output_dir, filename)
        tmp_path = final_path + ".tmp"

        # Skip if already complete
        if os.path.exists(final_path):
            return f"✔ Skipped (already done): {filename}"

        df = pd.read_parquet(os.path.join(input_dir, filename))

        df['fr_x'] = df['from'].map(sites_map).fillna(-1).astype(int)
        df['do_x'] = df['domain'].map(sites_map).fillna(-1).astype(int)
        df['to_x'] = df['to'].map(sites_map).fillna(-1).astype(int)

        session_window = 9  # minutes
        df['delta'] = df[['user', 'time']].groupby('user').diff() / 60
        df['session_start'] = df['delta'] > session_window
        df['session'] = df[['user', 'session_start']].groupby('user').cumsum()

        df.to_parquet(tmp_path)
        os.rename(tmp_path, final_path)

        return f"✅ Completed: {filename}"

    except Exception as e:
        error_msg = f"[ERROR] {filename}:\n{traceback.format_exc()}"
        with open('errors.log', 'a') as log:
            log.write(error_msg + "\n")
        return f"❌ Failed: {filename}"

# --------- Main Logic --------- #
def main():
    all_files = os.listdir(input_dir)
    done_files = os.listdir(output_dir)
    todo_files = list(set(all_files) - set(done_files))
    todo_files.sort()

    print(f"🧮 {len(todo_files)} files remaining out of {len(all_files)}")

    POOL_SIZE = min(len(todo_files), mp.cpu_count() - 2, 8)  
    
    with mp.Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_partition_element, todo_files):
            print(result, flush=True)

if __name__ == '__main__':
    main()
