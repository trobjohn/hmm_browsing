import os
import gzip
import pandas as pd
from multiprocessing import Pool

RAW_DATA_DIR = './t_hmm_2023-10-03/'
RIVANNA_PATH = '/scratch/trj2j/hmm/int/'
ROLLODEX_PATH = './int/rolodex_u.csv'
COMPLETED_PATH = './int/completed.csv'
TARGET_DOMAINS_PATH = './data/news.csv'
POOL_SIZE = 10

# Load target domains
target_set = set(pd.read_csv(TARGET_DOMAINS_PATH)['domain'])

# Load list of already-completed files
try:
    completed = set(pd.read_csv(COMPLETED_PATH)['file'])
except FileNotFoundError:
    completed = set()

def process_file(file):
    if not file.endswith(".csv.gz") or file in completed:
        return []

    output_users = []

    with gzip.open(os.path.join(RAW_DATA_DIR, file), 'rt') as f:
        df = pd.read_csv(f, low_memory=False)

    df = df.drop(['tdate', 'category', 'subcategory', 'refer_domain'], axis=1)
    df = df.rename({'timestamp_ss2k': 'time'}, axis=1)
    df['from'] = pd.concat([pd.Series(['']), df['domain'][:-1]], ignore_index=True)
    df['to'] = pd.concat([df['domain'][1:], pd.Series([''])], ignore_index=True)
    df['dt'] = df['time'].diff()

    for u in df['person_id'].unique():
        df_u = df[df['person_id'] == u].drop(columns='person_id')
        if not target_set.intersection(df_u['domain'].unique()):
            continue

        output_users.append(u)
        user_file = os.path.join(RIVANNA_PATH, f'u_paths/path_{u}.csv')
        if os.path.exists(user_file):
            df_u.to_csv(user_file, mode='a', header=False, index=False)
        else:
            df_u.to_csv(user_file, index=False)

    return [(file, output_users)]

if __name__ == '__main__':
    files = os.listdir(RAW_DATA_DIR)
    pool = Pool(POOL_SIZE)
    results = pool.map(process_file, files)
    pool.close()
    pool.join()

    # Flatten and write metadata
    with open(COMPLETED_PATH, 'a') as cf, open(ROLLODEX_PATH, 'a') as uf:
        if os.stat(ROLLODEX_PATH).st_size == 0:
            uf.write('user\n')  # write header if file is new/empty
        for file_entry, users in [r[0] for r in results if r]:
            cf.write(file_entry + '\n')
            for u in users:
                uf.write(str(u) + '\n')
