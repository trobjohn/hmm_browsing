import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix

# Parameters
POOL_SIZE = 12
LOAD_SIZE = 20
PROPORTION = 0.025
TOP_SITE_LIMIT = 100
RIVANNA_PATH = '/scratch/trj2j/hmm/int/'
RESULT_PATH = os.path.join(RIVANNA_PATH, 'h', 'h_')


def load_focus_and_sites():
    rolo_s = pd.read_csv('./int/rolodex_sites.csv')['sites'].tolist()
    #rolo_u = pd.read_csv('./int/rolodex_u.csv')['user'].tolist()
    rolo_u = pd.read_csv('./int/rolodex_u.csv', header=None)[0].tolist()
    target = pd.read_csv('./data/news.csv')['domain']
    top = pd.read_csv('./data/top_sites.csv')['domain'].iloc[:TOP_SITE_LIMIT]
    focus = pd.concat([target, top]).drop_duplicates().tolist()
    rolo_s = list(set(rolo_s) - set(focus))
    return rolo_s, rolo_u, focus


def sample_users(rolo_u):
    np.random.seed(632623)
    sample_size = int(len(rolo_u) * PROPORTION)
    return list(np.random.choice(rolo_u, size=sample_size, replace=False))


def load_blocks(num_blocks, shape):
    J, K = shape
    h_out = lil_matrix((J, K), dtype=int)
    for i in range(num_blocks):
        with open(os.path.join(RESULT_PATH, f'out_{i}.pickle'), 'rb') as f:
            h_out_i = pickle.load(f)
        h_out += h_out_i
        print(f'Completed {i + 1} of {num_blocks} blocks ({(i + 1) / num_blocks:.1%})')
    return h_out


def main():
    rolo_s, rolo_u, focus = load_focus_and_sites()
    K = len(focus)
    J = len(rolo_s)

    # Save cleaned site list
    pickle.dump(rolo_s, open('./int/rolo_s_clean.pickle', 'wb'))

    rolo_u_sample = sample_users(rolo_u)
    N_blocks = len(rolo_u_sample) // LOAD_SIZE

    print(f'Total users sampled: {len(rolo_u_sample)}')
    print(f'Total blocks: {N_blocks}')

    h_out = load_blocks(N_blocks, shape=(J, K))
    H_out = h_out.tocsr()
    del h_out

    os.makedirs('./int', exist_ok=True)
    with open('./int/H_out.sav', 'wb') as f:
        pickle.dump(H_out, f)

    print('H_out embedding saved successfully.')


if __name__ == '__main__':
    main()
