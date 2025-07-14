import os
import pickle
import numpy as np
from scipy.sparse import hstack
from scipy.sparse.linalg import svds

# Parameters
H_PATH = '/scratch/trj2j/hmm/int/'
OUT_PATH = './int'
SVD_K = 10

def run_sparse_svd(H, k=SVD_K):
    print(f'Running sparse SVD on matrix of shape {H.shape} with k={k}')
    H = H.astype(np.float32)  # Ensure float type
    U, S, Vt = svds(H, k=k)
    idx = np.argsort(-S)
    U, S = U[:, idx], S[idx]
    embedding = U @ np.diag(S)
    return embedding

def main():
    # Load H_in and H_out from previous step
    with open(os.path.join(H_PATH, 'H_in.sav'), 'rb') as f:
        H_in = pickle.load(f)
    with open(os.path.join(H_PATH, 'H_out.sav'), 'rb') as f:
        H_out = pickle.load(f)

    # Combine them into a single matrix
    H_combined = hstack([H_in, H_out])
    print(f'Combined shape: {H_combined.shape}')

    # Optional: save combined H
    with open(os.path.join(OUT_PATH, 'H_combined.sav'), 'wb') as f:
        pickle.dump(H_combined, f)

    # Run SVD
    embedding = run_sparse_svd(H_combined)

    # Save embedding
    with open(os.path.join(OUT_PATH, 'page_embed_combined.sav'), 'wb') as f:
        pickle.dump(embedding, f)

    print('Page embedding saved successfully.')

if __name__ == '__main__':
    main()
