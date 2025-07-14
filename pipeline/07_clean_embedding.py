import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, hstack
from scipy.sparse.linalg import svds

# Parameters
POOL_SIZE = 12
LOAD_SIZE = 20
PROPORTION = 0.025  # Set to 1.0 to use all users
TOP_SITE_LIMIT = 100
RIVANNA_PATH = '/scratch/trj2j/hmm/'
RESULT_PATH = os.path.join(RIVANNA_PATH, 'h', 'h_')


def load_focus_and_sites():
    rolo_s = pd.read_csv('./sweep/rolodex_sites.csv')['sites'].tolist()
    rolo_u = pd.read_csv('./sweep/rolodex_u.csv')['user'].tolist()
    target = pd.read_csv('./data/news.csv')['domain']
    top = pd.read_csv('./data/top_sites.csv')['domain'].iloc[:TOP_SITE_LIMIT]
    focus = pd.concat([target, top]).drop_duplicates().tolist()
    rolo_s = list(set(rolo_s) - set(focus))
    return rolo_s, rolo_u, focus


def sample_users(rolo_u, proportion=PROPORTION):
    if proportion >= 1.0:
        return rolo_u
    np.random.seed(632623)
    sample_size = int(len(rolo_u) * proportion)
    return list(np.random.choice(rolo_u, size=sample_size, replace=False))


def load_blocks(num_blocks, shape):
    J, K = shape
    h_in = lil_matrix((J, K), dtype=int)
    h_out = lil_matrix((J, K), dtype=int)
    for i in range(num_blocks):
        with open(os.path.join(RESULT_PATH, f'in_{i}.pickle'), 'rb') as f:
            h_in_i = pickle.load(f)
        with open(os.path.join(RESULT_PATH, f'out_{i}.pickle'), 'rb') as f:
            h_out_i = pickle.load(f)
        h_in += h_in_i
        h_out += h_out_i
        print(f'Completed {i + 1} of {num_blocks} blocks ({(i + 1) / num_blocks:.1%})')
    return h_in, h_out


def run_sparse_svd(H, k=10):
    print(f'Running sparse SVD on matrix of shape {H.shape} with k={k}')
    U, S, Vt = svds(H, k=k)
    idx = np.argsort(-S)
    U, S, Vt = U[:, idx], S[idx], Vt[idx, :]
    embedding = U @ np.diag(S)
    return embedding


def main():
    rolo_s, rolo_u, focus = load_focus_and_sites()
    K = len(focus)
    J = len(rolo_s)

    # Save cleaned site list
    pickle.dump(rolo_s, open('./sweep/rolo_s_clean.pickle', 'wb'))

    rolo_u_sample = sample_users(rolo_u, PROPORTION)
    N_blocks = len(rolo_u_sample) // LOAD_SIZE

    print(f'Total users sampled: {len(rolo_u_sample)}')
    print(f'Total blocks: {N_blocks}')

    h_in, h_out = load_blocks(N_blocks, shape=(J, K))
    H_in = h_in.tocsr().astype(float)
    H_out = h_out.tocsr().astype(float)
    del h_in, h_out

    H = hstack([H_in, H_out])
    del H_in, H_out

    os.makedirs('./sweep', exist_ok=True)
    with open('./sweep/H_combined.sav', 'wb') as f:
        pickle.dump(H, f)

    # Run SVD and save embedding
    embedding = run_sparse_svd(H, k=10)
    with open('./sweep/page_embed_combined.sav', 'wb') as f:
        pickle.dump(embedding, f)

    print('Combined H_in + H_out SVD embedding saved successfully.')


if __name__ == '__main__':
    main()
