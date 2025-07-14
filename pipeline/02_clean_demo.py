import pandas as pd
from pathlib import Path

df = pd.read_csv('./data/hmm_demos_2023-10-03.csv', low_memory=False)
path = Path('./int/u_cat/')
path.mkdir(exist_ok=True)

aggregate_stats = []

# Group by actual observed combinations
counter = 0
for keys, group in df.groupby(['gender', 'race_id', 'ethnicity_id']):
    gen, race, eth = keys
    hits = group.shape[0]
    if hits > 0:
        fn = path / f'cat_{counter}.csv'
        group.to_csv(fn, index=False)
        aggregate_stats.append([counter, hits, gen, race, eth])
        counter += 1

# Save metadata
agg = pd.DataFrame(aggregate_stats, columns=['counter', 'size', 'gen_code', 'race_code', 'eth_code'])
agg.sort_values('size', ascending=False, ignore_index=True, inplace=True)
agg.to_csv('./int/u_groups.csv', index=False)

