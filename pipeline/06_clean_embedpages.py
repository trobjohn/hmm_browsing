import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
from multiprocessing import Pool, cpu_count

# Paths
RIVANNA_PATH = '/scratch/trj2j/hmm/int/'
ALLOC_PATH = os.path.join(RIVANNA_PATH, 'site_allocations')
U_PATH = os.path.join(RIVANNA_PATH, 'u_paths')
RESULT_PATH = os.path.join(RIVANNA_PATH, 'h')
os.makedirs(RESULT_PATH, exist_ok=True)

# Parameters
POOL_SIZE = cpu_count()
MAX_FILE_SIZE = 3_000_000  # bytes

def process_u_block(block_number):
    print(f'Processing block {block_number}', flush=True)
    assignment_path = os.path.join(ALLOC_PATH, f'assignment_{block_number}.pickle')
    with open(assignment_path, 'rb') as f:
        assignment = pickle.load(f)

    this_block = assignment['this_block']
    focus = assignment['focus']
    rolo_s = assignment['rolo_s']
    K, J = len(focus), len(rolo_s)

    focus_index = {label: idx for idx, label in enumerate(focus)}
    rolo_s_index = {label: idx for idx, label in enumerate(rolo_s)}

    h_in_block = dok_matrix((J, K), dtype=int)
    h_out_block = dok_matrix((J, K), dtype=int)

    these_paths = pd.DataFrame()
    for u in this_block:
        file_path = os.path.join(U_PATH, f'path_{u}.csv')
        if os.path.exists(file_path) and os.stat(file_path).st_size < MAX_FILE_SIZE:
            try:
                df = pd.read_csv(file_path, usecols=['domain', 'from', 'to'], low_memory=False)
                these_paths = pd.concat([these_paths, df], axis=0)
            except Exception as e:
                print(f'Warning: could not read {file_path}: {e}', flush=True)

    if these_paths.empty:
        print(f'No usable data in block {block_number}', flush=True)
        return block_number

    # ARRIVALS
    arrivals = pd.crosstab(these_paths['domain'], these_paths['from'])
    arr = arrivals.loc[arrivals.index.intersection(focus), arrivals.columns.difference(focus)]
    for row_label in arr.index:
        c = focus_index.get(row_label)
        for col_label in arr.columns:
            r = rolo_s_index.get(col_label)
            if r is not None and c is not None:
                h_in_block[r, c] += arr.at[row_label, col_label]

    # DEPARTURES
    departures = pd.crosstab(these_paths['domain'], these_paths['to'])
    dep = departures.loc[departures.index.intersection(focus), departures.columns.difference(focus)]
    for row_label in dep.index:
        c = focus_index.get(row_label)
        for col_label in dep.columns:
            r = rolo_s_index.get(col_label)
            if r is not None and c is not None:
                h_out_block[r, c] += dep.at[row_label, col_label]

    # Save results
    with open(os.path.join(RESULT_PATH, f'in_{block_number}.pickle'), 'wb') as f:
        pickle.dump(h_in_block, f)
    with open(os.path.join(RESULT_PATH, f'out_{block_number}.pickle'), 'wb') as f:
        pickle.dump(h_out_block, f)

    return block_number

def main():
    print('Counting assignment blocks...')
    num_blocks = sum(1 for f in os.listdir(ALLOC_PATH) if f.startswith('assignment_'))
    print(f'Found {num_blocks} blocks. Starting processing...')

    with Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_u_block, range(num_blocks)):
            print(f'Completed block {result}', flush=True)

if __name__ == '__main__':
    main()
