import os
import pickle
import pandas as pd
import itertools
from multiprocessing import Pool

rivanna_path = '/scratch/trj2j/hmm/'
POOL_SIZE = 20


def get_sites(file):
    df = pd.read_csv(os.path.join(rivanna_path, 'u_paths', file), low_memory=False)
    # Union of domain, from, and to columns
    return set(df['domain']).union(df['from']).union(df['to'])


def clean_sites(site_set):
    # Remove nulls, cast all to string, remove leading/trailing whitespace
    return sorted(set(str(s).strip() for s in site_set if pd.notnull(s)))


if __name__ == '__main__':
    print('Creating sitelist...')

    files = os.listdir(os.path.join(rivanna_path, 'u_paths'))
    gdf = pd.DataFrame({'paths': files})
    gdf.to_csv('./int/all_paths.csv', index=False)

    with Pool(POOL_SIZE) as pool:
        results = pool.map(get_sites, files)

    print('Merging sets to list and saving...')
    merged = set(itertools.chain.from_iterable(results))
    cleaned_sites = clean_sites(merged)

    os.makedirs('./int', exist_ok=True)
    pd.DataFrame({'sites': cleaned_sites}).to_csv('./int/rolodex_sites.csv', index=False)
    print(f"Saved {len(cleaned_sites)} cleaned site names to ./int/rolodex_sites.csv")
