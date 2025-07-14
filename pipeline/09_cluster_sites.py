import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
import os

# Constants
EMBED_PATH = './int/page_embed_combined.sav'
SITE_PATH = './int/rolo_s_clean.pickle'
CLUSTERS_CSV = './int/kmc_clusters.csv'
MODEL_OUT = './int/kmc.sav'
SITES_MAP_PATH = './int/sites_map.sav'
N_CLUSTERS = 500
SEED = 0

def main():
    # Load data
    with open(SITE_PATH, 'rb') as f:
        domains = pickle.load(f)
    with open(EMBED_PATH, 'rb') as f:
        X = pickle.load(f)

    # Normalize across each dimension
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    print('Embedding shape:', X.shape)

    # Clustering
    print('Running KMeans clustering...')
    kmc = KMeans(n_clusters=N_CLUSTERS, max_iter=1000, random_state=SEED, n_init='auto')
    kmc.fit(X)

    # Generate cluster dataframe with canonical site index
    cluster_df = pd.DataFrame({
        'domain': domains,
        'cluster': kmc.labels_,
        'site_index': np.arange(len(domains))
    })

    # Save results
    cluster_df.to_csv(CLUSTERS_CSV, index=False)
    pickle.dump(kmc, open(MODEL_OUT, 'wb'))

    # Save mapping as a Series
    sites_map = pd.Series(data=cluster_df['site_index'].values, index=cluster_df['domain'].values)
    pickle.dump(sites_map, open(SITES_MAP_PATH, 'wb'))

    print('✅ KMeans clustering, mapping, and model saved.')

if __name__ == '__main__':
    main()
