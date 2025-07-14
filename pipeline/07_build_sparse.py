import os
import pickle
import argparse
import pandas as pd
from scipy.sparse import lil_matrix
from multiprocessing import Pool, cpu_count

# Constants
TOP_SITE_LIMIT = 100
RIVANNA_PATH = '/scratch/trj2j/hmm/int/'
RESULT_PATH = os.path.join(RIVANNA_PATH, 'h')
ROLO_S_CLEAN_PATH = './int/rolo_s_clean.pickle'


def load_focus_and_sites():
    rolo_s = pd.read_csv('./int/rolodex_sites.csv')['sites'].tolist()
    target = pd.read_csv('./data/news.csv')['domain']
    top = pd.read_csv('./data/top_sites.csv')['domain'].iloc[:TOP_SITE_LIMIT]
    focus = pd.concat([target, top]).drop_duplicates().tolist()
    rolo_s = list(set(rolo_s) - set(focus))
    return rolo_s, focus


def load_single_block(args):
    polarity, i, shape = args
    J, K = shape
    file_path = os.path.join(RESULT_PATH, f'{polarity}_{i}.pickle')
    with open(file_path, 'rb') as f:
        h_i = pickle.load(f)
    return h_i


def load_blocks_parallel(polarity, num_blocks, shape):
    print(f"Using {min(cpu_count(), 12)} cores for loading blocks...")
    with Pool(processes=min(cpu_count(), 12)) as pool:
        blocks = pool.map(load_single_block, [(polarity, i, shape) for i in range(num_blocks)])
    h = lil_matrix(shape, dtype=int)
    for i, h_i in enumerate(blocks):
        h += h_i
        print(f'Loaded {i + 1}/{num_blocks} blocks ({(i + 1)/num_blocks:.1%})')
    return h


def process(polarity):
    rolo_s, focus = load_focus_and_sites()
    J, K = len(rolo_s), len(focus)

    # Save cleaned site list once, with atomic write to avoid race conditions
    if polarity == 'in':
        try:
            with open(ROLO_S_CLEAN_PATH, 'xb') as f:  # Atomic write, fail if file exists
                pickle.dump(rolo_s, f)
        except FileExistsError:
            print("✔ rolo_s_clean.pickle already exists; skipping write.")

    files = [f for f in os.listdir(RESULT_PATH) if f.startswith(f'{polarity}_') and f.endswith('.pickle')]
    N_blocks = len(files)
    print(f'Polarity = {polarity}, Blocks found: {N_blocks}')

    h = load_blocks_parallel(polarity, N_blocks, (J, K))
    H = h.tocsr()
    del h

    with open(os.path.join(RIVANNA_PATH, f'H_{polarity}.sav'), 'wb') as f:
        pickle.dump(H, f)
    print(f'H_{polarity} saved successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate embed pages into sparse matrix")
    parser.add_argument('--polarity', choices=['in', 'out'], required=True, help="Whether to process 'in' or 'out' links")
    args = parser.parse_args()
    process(args.polarity)
