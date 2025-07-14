import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

# Configuration
POOL_SIZE = 12
LOAD_SIZE = 20
TOP_SITE_LIMIT = 100
RIVANNA_PATH = '/scratch/trj2j/hmm/int/'
RESULT_PATH = os.path.join(RIVANNA_PATH, 'h')
POLARITY = 'in'  # Change to 'out' for H_out aggregation

def load_focus_and_sites():
    """Load and clean site lists to get rolo_s and focus."""
    rolo_s = pd.read_csv('./int/rolodex_sites.csv')['sites'].tolist()
    target = pd.read_csv('./data/news.csv')['domain']
    top = pd.read_csv('./data/top_sites.csv')['domain'].iloc[:TOP_SITE_LIMIT]
    focus = pd.concat([target, top]).drop_duplicates().tolist()
    rolo_s = list(set(rolo_s) - set(focus))
    return rolo_s, focus

def load_blocks(num_blocks, shape):
    """Load and sum sparse matrices from disk."""
    J, K = shape
    h = lil_matrix((J, K), dtype=int)
    for i in range(num_blocks):
        path = os.path.join(RESULT_PATH, f'{POLARITY}_{i}.pickle')
        with open(path, 'rb') as f:
            h_i = pickle.load(f)
        h += h_i
        print(f'Completed {i + 1} of {num_blocks} blocks ({(i + 1) / num_blocks:.1%})')
    return h

def main():
    rolo_s, focus = load_focus_and_sites()
    K = len(focus)
    J = len(rolo_s)

    # Save cleaned idiosyncratic site list
    with open('./int/rolo_s_clean.pickle', 'wb') as f:
        pickle.dump(rolo_s, f)

    # Get list of matching pickles
    files = [f for f in os.listdir(RESULT_PATH) if f.startswith(POLARITY + '_') and f.endswith('.pickle')]
    N_blocks = len(files)
    print(f'Total blocks: {N_blocks}')

    h = load_blocks(N_blocks, shape=(J, K))
    H = h.tocsr()
    del h  # Free memory

    output_path = os.path.join(RIVANNA_PATH, f'H_{POLARITY}.sav')
    with open(output_path, 'wb') as f:
        pickle.dump(H, f)

    print(f'H_{POLARITY} embedding saved successfully.')

if __name__ == '__main__':
    main()
