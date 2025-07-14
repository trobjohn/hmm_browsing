import numpy as np
import pandas as pd
from multiprocessing import Pool
import os

POOL_SIZE = 23
LOAD_SIZE = 100
SLOTS = 500
RIVANNA_PATH = '/scratch/trj2j/hmm/'

def process_block(files):
    counts = []
    for path in files:
        df = pd.read_csv(os.path.join(RIVANNA_PATH, 'u_paths', path), low_memory=False)
        tab = df['domain'].value_counts()
        queue = pd.DataFrame({'domain': tab.index, 'hits': tab.values})
        queue = queue[~queue['domain'].isin(target_set)]
        counts.append(queue)

    merged = pd.concat(counts, ignore_index=True)
    reduced = (
        merged
        .groupby('domain', as_index=False)['hits']
        .sum()
        .sort_values('hits', ascending=False)
        .head(SLOTS)
    )
    return reduced

def consolidate(dfs):
    merged = pd.concat(dfs, ignore_index=True)
    return (
        merged
        .groupby('domain', as_index=False)['hits']
        .sum()
        .sort_values('hits', ascending=False)
        .head(SLOTS)
    )

if __name__ == '__main__':
    print("Loading path list and target domains...")
    u_paths = pd.read_csv('./int/all_paths.csv', low_memory=False)['paths'].tolist()
    target = pd.read_csv('./data/news.csv')
    target_set = set(target['domain'])

    I = len(u_paths)
    N_blocks = I // LOAD_SIZE
    u_blocks = np.array(u_paths[:N_blocks * LOAD_SIZE]).reshape(N_blocks, LOAD_SIZE)
    u_remainder = u_paths[N_blocks * LOAD_SIZE:]

    print(f"Processing {N_blocks} blocks of {LOAD_SIZE} users each...")

    with Pool(POOL_SIZE) as pool:
        results = pool.map(process_block, u_blocks)

    print("Processing remainder block...")
    remainder_result = process_block(u_remainder)

    print("Consolidating results...")
    results.append(remainder_result)
    top_sites = consolidate(results)

    top_sites.to_csv('./int/top_sites.csv', index=False)
    print("Saved top sites to ./int/top_sites.csv")
