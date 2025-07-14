import numpy as np
import pandas as pd
from multiprocessing import Pool


def process_block(files):
    ##  Initialize
    #files = u_blocks[k,:]
    ##  Initialize the list
    df = pd.read_csv('/scratch/trj2j/hmm/u_paths/'+files[0])
    tab = df['domain'].value_counts()
    queue = pd.DataFrame({'domain':tab.index, 'hits':tab.values})
    drop_these = queue['domain'].isin(target_set)
    queue = queue[drop_these==False]

    # Process the rest of the block
    for j in range(1,len(files)):
        path = files[j]
        df = pd.read_csv('/scratch/trj2j/hmm/u_paths/'+path,low_memory=False)
        tab = df['domain'].value_counts()
        queue_i = pd.DataFrame({'domain':tab.index, 'hits':tab.values})
        drop_these = queue_i['domain'].isin(target_set)
        queue_i = queue_i[drop_these==False]
        L1 = queue['domain'].to_list()
        L2 = queue_i['domain'].to_list()
        from_2_to_1 = [L1.index(i) for i in L2 if i in L1]
        from_1_to_2 = [L2.index(i) for i in L1 if i in L2]
        # Update list
        queue.iloc[ from_2_to_1, queue.columns.get_loc('hits')] = queue.iloc[ from_2_to_1,:]['hits'].to_numpy()+\
            (queue_i.iloc[ from_1_to_2,:]['hits']).to_numpy()
        # Eliminate duplicate values from queue_i
        to_drop = [L2[i] for i in from_1_to_2]
        select = queue_i['domain'].isin(to_drop)
        queue_i = queue_i[  select==False]
        # Truncate back to 'slots'
        queue = pd.concat([queue,queue_i],axis=0, ignore_index=True)
        if queue.shape[0] > SLOTS:
            queue.sort_values('hits',inplace=True,ignore_index=True,ascending=False)    
            queue = queue.iloc[:SLOTS,]
    return queue

def consolidate(result):
    hits = result[0]
    for j in range(1,len(result)):
        hits_j = result[j]
        #
        L1 = hits['domain'].to_list()
        L2 = hits_j['domain'].to_list()
        from_2_to_1 = [L1.index(i) for i in L2 if i in L1]
        from_1_to_2 = [L2.index(i) for i in L1 if i in L2]
        #
        # Update hits
        hits.iloc[ from_2_to_1, hits.columns.get_loc('hits')] = hits.iloc[ from_2_to_1,:]['hits'].to_numpy()+\
            (hits_j.iloc[ from_1_to_2,:]['hits']).to_numpy()
        # Eliminate duplicate values from queue_i
        to_drop = [L2[i] for i in from_1_to_2]
        select = hits_j['domain'].isin(to_drop)
        hits_j = hits_j[  select==False]
        # Truncate back to 'slots'
        hits = pd.concat([hits,hits_j],axis=0, ignore_index=True)
        if hits.shape[0] > SLOTS:
            hits.sort_values('hits',inplace=True,ignore_index=True,ascending=False)    
            hits = hits.iloc[:SLOTS,]
    return hits

if __name__ == '__main__':

    POOL_SIZE = 23
    LOAD_SIZE = 100
    SLOTS = 500

    u_paths = pd.read_csv('./sweep/all_paths.csv',low_memory=False)
    u_paths = u_paths['paths'].to_list()
    I = len(u_paths)

    target = pd.read_csv('./data/news.csv')
    target_set = target['domain']
    J = len(target_set)
            
    N_blocks = int( np.floor(I/LOAD_SIZE) )
    u_blocks = np.array(u_paths[:N_blocks*LOAD_SIZE]).reshape(N_blocks,LOAD_SIZE)
    u_remainder = u_paths[N_blocks*LOAD_SIZE:]

    print('Pooling')
    POOL = Pool(POOL_SIZE)
    results = POOL.map_async(process_block,u_blocks)
    POOL.close()
    POOL.join()
    # Forgot to do the remainder sites
    results_remainder = process_block(u_remainder)


    #top_sites = consolidate([results,results_remainder])
    print('Consolidating')
    #top_sites = consolidate(results.get())
    result = results.get()
    result.append(results_remainder)
    hits = result[0]
    for j in range(1,len(result)):
        hits_j = result[j]
        #
        L1 = hits['domain'].to_list()
        L2 = hits_j['domain'].to_list()
        from_2_to_1 = [L1.index(i) for i in L2 if i in L1]
        from_1_to_2 = [L2.index(i) for i in L1 if i in L2]
        #
        # Update hits
        hits.iloc[ from_2_to_1, hits.columns.get_loc('hits')] = hits.iloc[ from_2_to_1,:]['hits'].to_numpy()+\
            (hits_j.iloc[ from_1_to_2,:]['hits']).to_numpy()
        # Eliminate duplicate values from queue_i
        to_drop = [L2[i] for i in from_1_to_2]
        select = hits_j['domain'].isin(to_drop)
        hits_j = hits_j[  select==False]
        # Truncate back to 'slots'
        hits = pd.concat([hits,hits_j],axis=0, ignore_index=True)
        if hits.shape[0] > SLOTS:
            hits.sort_values('hits',inplace=True,ignore_index=True,ascending=False)    
            hits = hits.iloc[:SLOTS,]
    top_sites = hits
    top_sites.to_csv('./data/top_sites.csv',index=False)
