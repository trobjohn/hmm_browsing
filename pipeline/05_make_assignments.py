import os
import pickle
import pandas as pd
import itertools
import random
from multiprocessing import Pool

rivanna_path = '/scratch/trj2j/hmm/int/'
POOL_SIZE = 20
FULL_SAMPLE = False  # Toggle to True for full user list
PROPORTION = 0.025   # Used only if FULL_SAMPLE is False
TOP_SITE_LIMIT = 100
LOAD_SIZE = 20

rng = random.Random(632623)

def get_sites(file):
    df = pd.read_csv(os.path.join(rivanna_path, 'u_paths', file), low_memory=False)
    return set(df['domain']).union(df['from']).union(df['to'])

def clean_sites(site_set):
    return sorted(set(str(s).strip() for s in site_set if pd.notnull(s) and str(s).strip()))

def make_assignments(rolo_u, rolo_s, focus, load_size, out_path):
    u_blocks = [rolo_u[k*load_size : (k+1)*load_size] 
                for k in range((len(rolo_u) + load_size - 1) // load_size)]
    os.makedirs(out_path, exist_ok=True)
    for k, block in enumerate(u_blocks):
        assignment = {'focus': focus, 'rolo_s': rolo_s, 'this_block': block}
        assignment_path = os.path.join(out_path, f'assignment_{k}.pickle')
        if not os.path.exists(assignment_path):
            with open(assignment_path, 'wb') as f:
                pickle.dump(assignment, f)
    print(f"Wrote {len(u_blocks)} assignment files to {out_path}")

if __name__ == '__main__':
    print('Creating sitelist...')

    files = os.listdir(os.path.join(rivanna_path, 'u_paths'))
    gdf = pd.DataFrame({'paths': files})

    with Pool(POOL_SIZE) as pool:
        results = pool.map(get_sites, files)

    print('Merging sets to list and saving...')
    merged = set(itertools.chain.from_iterable(results))
    cleaned_sites = clean_sites(merged)

    os.makedirs('./int', exist_ok=True)
    pd.DataFrame({'sites': cleaned_sites}).to_csv('./int/rolodex_sites.csv', index=False)
    print(f"Saved {len(cleaned_sites)} cleaned site names to ./int/rolodex_sites.csv")

    generate_assignments = True
    if generate_assignments:
        print("Generating assignments...")

        rolo_s = cleaned_sites
        rolo_u = pd.read_csv('./int/rolodex_u.csv', header=None)[0].tolist()
        target = pd.read_csv('./data/news.csv')['domain']
        top = pd.read_csv('./data/top_sites.csv')['domain'].iloc[:TOP_SITE_LIMIT]
        focus = pd.concat([target, top]).drop_duplicates().tolist()
        rolo_s = list(set(rolo_s) - set(focus))

        if not FULL_SAMPLE:
            rolo_u = rng.sample(rolo_u, int(len(rolo_u) * PROPORTION))
        pd.DataFrame({'user': rolo_u}).to_csv('./int/rolodex_u_sample.csv', index=False)

        make_assignments(rolo_u, rolo_s, focus, LOAD_SIZE, os.path.join(rivanna_path, 'site_allocations'))
