#!/bin/bash

set -e  # die on first failure
LOGFILE="pipeline_$(date +%Y%m%d_%H%M).log"

source .hmm/bin/activate

#echo "Starting full clean at $(date)" | tee -a $LOGFILE

### Step 1: Clean raw data
#echo "[1] Cleaning raw input files..." | tee -a $LOGFILE
#python 01_clean_paths.py
#python 02_clean_demo.py
#python 03_clean_sitelist.py
#python 04_clean_topsites.py
#python 05_make_assignments.py

## Step 2: Build embeddings
#echo "[2] Building embeddings..." | tee -a $LOGFILE
#python 06_clean_embedpages.py
#python 07_build_sparse.py --polarity in &
#python 07_build_sparse.py --polarity out &
#wait
#python 08_run_SVD.py

## Step 3: Clustering
#echo "[3] Clustering pages..." | tee -a $LOGFILE
#python 09_cluster_sites.py

### Step 4: Recode path data
#echo "[4] Recoding user paths..." | tee -a $LOGFILE
#python 10_recode_paths_to_buckets.py
#python 10.5_retry_recode_hmm_paths.py

### Step 5: Partition users by demographic data
#echo "[5] Partitioning by demographic data..." | tee -a $LOGFILE
python 11_consolidate_partitions.py

### Step 6: Partition users by demographic data
#echo "[6] Final cleanup for HMM input..." | tee -a $LOGFILE
python 12_clean_hmm_paths.py



echo "Finished at $(date)" | tee -a $LOGFILE
