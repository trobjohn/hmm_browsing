import os
import pandas as pd
import numpy as np
import traceback
import pickle
from pathlib import Path

# --------- Paths --------- #
INPUT_DIR = Path('/scratch/trj2j/hmm/int/u_paths/')
OUTPUT_DIR = Path('/scratch/trj2j/hmm/int/u_paths_recode/')

ROLO_S_PATH = './int/rolo_s_clean.pickle'
CLUSTERS_PATH = './int/kmc.sav'
CLUSTERS_CSV = './int/kmc_clusters.csv'
TOP_SITES_PATH = './data/top_sites.csv'
TARGET_SITES_PATH = './data/news.csv'


# --------- Load Mappings --------- #
print("🔍 Loading mappings...")
rolo_s_clean = pickle.load(open(ROLO_S_PATH, 'rb'))
#adf = pickle.load(open(CLUSTERS_PATH, 'rb'))
adf = pd.read_csv(CLUSTERS_CSV)
top = pd.read_csv(TOP_SITES_PATH)
target = pd.read_csv(TARGET_SITES_PATH)

target_sites = target['domain']
top_sites = top['domain']
clusters = adf['cluster'].unique()
sites = [None] + target_sites.tolist() + top_sites.tolist() + clusters.astype(str).tolist()
sites_map = pd.Series(np.arange(len(sites)), index=sites)

# --------- Step 1: Scan for suspicious or corrupted files --------- #
print("🔎 Scanning output directory...")
retry_files = []

for f in OUTPUT_DIR.glob("*.parquet"):
    try:
        df = pd.read_parquet(f)
        if df.empty:
            print(f"⚠️ Empty file: {f.name}")
            retry_files.append(f.name)
    except Exception as e:
        print(f"❌ Failed to load: {f.name} ({e})")
        retry_files.append(f.name)

# --------- Optional: Remove orphaned .tmp files --------- #
for tmp_file in OUTPUT_DIR.glob("*.tmp"):
    final_file = OUTPUT_DIR / tmp_file.name.replace(".tmp", "")
    if not final_file.exists():
        print(f"🧹 Removing orphaned tmp: {tmp_file.name}")
        tmp_file.unlink()

# --------- Step 2: Retry Processing --------- #
def process_file(filename):
    try:
        final_path = OUTPUT_DIR / filename
        tmp_path = final_path.with_suffix(".parquet.tmp")

        if final_path.exists():
            print(f"✔ Skipping {filename} (already redone)")
            return

        print(f"🚧 Retrying {filename}...")
        df = pd.read_parquet(INPUT_DIR / filename)

        df['fr_x'] = df['from'].map(sites_map).fillna(-1).astype(int)
        df['do_x'] = df['domain'].map(sites_map).fillna(-1).astype(int)
        df['to_x'] = df['to'].map(sites_map).fillna(-1).astype(int)

        session_window = 9
        df['delta'] = df[['user', 'time']].groupby('user').diff() / 60
        df['session_start'] = df['delta'] > session_window
        df['session'] = df[['user', 'session_start']].groupby('user').cumsum()

        df.to_parquet(tmp_path, compression=None)
        os.replace(tmp_path, final_path)
        print(f"✅ Done: {filename}")

    except Exception as e:
        print(f"❌ Error on {filename}:\n{traceback.format_exc()}")
        with open('errors_retry_hmm_paths.log', 'a') as f:
            f.write(f"[{filename}]\n{traceback.format_exc()}\n\n")

if __name__ == "__main__":
    print(f"🔁 Retrying {len(retry_files)} files...")
    for file in retry_files:
        process_file(file)
