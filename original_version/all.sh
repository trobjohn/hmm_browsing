# for f in *; do rm "$f"; done
# ls -1 | wc -l


## Clean raw data files
#python clean_paths.py # REMEMBER TO CLEAR PATHS TO AVOID APPENDING
#python clean_demo.py
#python clean_sitelist.py 



## Page embedding
#python clean_topsites.py
#python clean_embedpages.py 
#    # Run clean_embedpages_cleanup.ipynb for deadlocked assignments
#python clean_embedpages_in.py # run simultaneously with _out
#python clean_embedpages_out.py # run simultaneously with _in
#python clean_embagedpages_svd.py # 91% of variance explained


## Cluster
#python clean_clustersites.py # Played in _d10 to make a scree plot and pick k; I deleted _d10, oh well.


## Recode original paths
#python clean_recodepaths.py # If interrupted, create_cleanup_list.ipynb, and uncomment the corresponding line
#python clean_consolidate_partitions.py # Good Lord what a mess. Expect to delete parquet files and restart the kernel.