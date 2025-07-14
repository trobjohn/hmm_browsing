import os

input_dir = '/scratch/trj2j/hmm/u_paths_partitioned/'
output_dir = '/scratch/trj2j/hmm/u_paths_hmm/'

done = set(f for f in os.listdir(output_dir) if not f.endswith('.tmp'))
remaining = sorted(set(os.listdir(input_dir)) - done)

print(f"{len(remaining)} remaining files:")
for f in remaining:
    print(f)

# Save list to file for reuse
with open("remaining_stragglers.txt", "w") as f:
    for r in remaining:
        f.write(r + "\n")