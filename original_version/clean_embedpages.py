import os as os
import pandas as pd
#import polars as pl
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import seaborn as sns
import pickle
import multiprocessing as mp
import random


def process_u_block(block_number):
    print('Processing block: ', block_number, flush=True)
    assignment = pickle.load(open('/scratch/trj2j/hmm/site_allocations/assignment_'+str(block_number)+'.pickle','rb'))
    this_block = assignment['this_block']
    focus = assignment['focus']
    K = len(focus)
    rolo_s = assignment['rolo_s']
    J = len(rolo_s)
    path = "/scratch/trj2j/hmm/u_paths/"

    ## Prep
    h_in_block = lil_matrix((J,K), dtype=int) # other, focus
    h_out_block = lil_matrix((J,K), dtype=int)

    # Get all users in the block at once
    these_paths = pd.DataFrame()
    for u in this_block:
        # Skip the really large path files    
        path = "/scratch/trj2j/hmm/u_paths/"
        file = 'path_'+str(u)+'.csv'
        this_size = os.stat(os.path.join(path,file)).st_size
        if this_size < 3000000: #4900000
            user_path = pd.read_csv(path+file,low_memory=False)
            these_paths = pd.concat([these_paths, user_path],axis=0)
    ######################################################################################################
    ## ARRIVAL ANALYSIS
    arrivals = pd.crosstab(these_paths['domain'],these_paths['from'])
    # Shrink arrival rows to focus
    row_names = arrivals.index.to_list()
    col_names = arrivals
    r_select = [ i in focus for i in row_names ] # Select only focus sites from rows of arrival
    c_select = [ i not in focus for i in col_names ] # Select only focus sites from rows of arrival
    arr = arrivals.iloc[r_select,c_select] # Shrink to focus sites
    row_names = arr.index.to_list()
    col_names = arr.columns
    # Locate in larger array
    focus_locate = [ focus.index(i) for i in row_names ] # focus sites
    idio_locate = [ rolo_s.index(i) for i in col_names] # idio sites -- This is the major time cost
    # Place values in h_:
    for f in range(len(focus_locate)): # focus
        c = focus_locate[f]
        for i in range(len(idio_locate)): # other
            r = idio_locate[i]
            h_in_block[ (r,c)] += arr.iloc[f,i] # Get this right!
    ######################################################################################################
    ## DEPARTURE ANALYSIS
    departures = pd.crosstab(these_paths['domain'],these_paths['to'])
    # Shrink arrival rows to focus
    row_names = departures.index.to_list()
    col_names = departures
    r_select = [ i in focus for i in row_names ] # Select only focus sites from rows of arrival
    c_select = [ i not in focus for i in col_names ] # Select only focus sites from rows of arrival
    dep = departures.iloc[r_select,c_select] # Shrink to focus sites
    row_names = dep.index.to_list()
    col_names = dep.columns
    # Locate in larger array
    focus_locate = [ focus.index(i) for i in row_names ] # focus sites
    idio_locate = [ rolo_s.index(i) for i in col_names] # idio sites -- This is the major time cost
    # Place values:
    for f in range(len(focus_locate)):
        c = focus_locate[f]
        for i in range(len(idio_locate)):
            r = idio_locate[i]
            h_out_block[ (r,c)] += dep.iloc[f,i] # Get this right!
    #print('Completed block: ', block_number, flush=True)
    ###
    result_path = '/scratch/trj2j/hmm/h/h_'
    pickle.dump(h_in_block, open(result_path+'in_'+str(block_number)+'.pickle', 'wb'))
    pickle.dump(h_out_block, open(result_path+'out_'+str(block_number)+'.pickle', 'wb'))
    return block_number


def main():

    print('Loading objects and data')

    POOL_SIZE = 20  # Rivanna cores, about 25
    LOAD_SIZE = 20
    PROPORTION = .025
    TOP_SITE_LIMIT = 100

    rivanna_path = '/scratch/trj2j/hmm/u_paths/'

    ## Sites and users:
    rolo_s = pd.read_csv('./sweep/rolodex_sites.csv')['sites']
    rolo_u = pd.read_csv('./sweep/rolodex_u.csv')['user'].to_list()

    ## Target sites: top sights and focus sights
    target = pd.read_csv('./data/news.csv')['domain']
    top = pd.read_csv('./data/top_sites.csv')['domain']
    top = top[:TOP_SITE_LIMIT]
    focus = pd.concat([target,top]) # The labels for k, focus sites
    rolo_s = list(set(rolo_s) - set(focus)) # The labels for j, idio sites
    focus = list(focus)
    #pickle.dump(rolo_s, open('./sweep/idio.pickle', 'wb'))
    #pickle.dump(focus, open('./sweep/focus.pickle', 'wb'))


    K = len(focus) # Number of focus sites
    J = len(rolo_s) # Total number of sites

    pickle.dump(rolo_s, open('./sweep/rolo_s_clean.pickle', 'wb')) #<- rolodex without focus sites
        #rolo_s = pickle.load(open('./models/rolo_s_clean.pickle','rb'))

    ## Take a large random sample of users. Otherwise this will never finish. //63623
    random.seed(632623)
    rolo_u = random.sample(rolo_u, int(len(rolo_u)*PROPORTION) )
    I = len(rolo_u) # Number of users

    ## Create sparse matrix version in lil format
    print('Creating sparse matrix representation of the data...')
    N_blocks = int( np.floor(I/LOAD_SIZE))
    print('Total blocks: ', N_blocks, '\n')
    #u_blocks = np.array(rolo_u[:N_blocks*LOAD_SIZE]).reshape(N_blocks,LOAD_SIZE)
    u_blocks= [ rolo_u[k*LOAD_SIZE : (k+1)*LOAD_SIZE] for k in range(int(len(rolo_u)/LOAD_SIZE))]

    N_blocks = len(u_blocks)

    site_path = '/scratch/trj2j/hmm/site_allocations/'

    build_blocks = 0
    if build_blocks == 1:
        for k in range(N_blocks):
            assignment = {'focus':focus,'rolo_s':rolo_s,'this_block':u_blocks[k]}
            pickle.dump(assignment, open(site_path+'assignment_'+str(k)+'.pickle', 'wb'))

#    h_in = lil_matrix((J, K), dtype=float) # (Idio sites, focus sites)
 #   h_out = lil_matrix((J, K), dtype=float) # (Idio Sites, focus Sites)

    #u_blocks= u_blocks[0:3]
#    POOL = mp.Pool(POOL_SIZE)
    



# #,mp_context=mp.get_context("spawn")
#     #mp.set_start_method("spawn")
#     with Pool(POOL_SIZE) as pool:
#         #for result in pool.imap_unordered(process_u_block, range(15)  ) :
#         for result in pool.imap_unordered( process_u_block, u_blocks  ) :            
#             h_in = h_in + result[0]
#             h_out = h_out + result[1]
#         #pool.close()
#         #pool.join()




    # with mp.Pool(POOL_SIZE) as pool:
    #     result = pool.map(process_u_block, u_blocks)
    #     for out in result.get():
    #         h_in = h_in + out[0]
    #         h_out = h_out + out[1]



    #result = [ process_u_block(t) for t in range(N_blocks)]
    # result = process_u_block(0)
    # print(result)



    # with mp.Pool(POOL_SIZE) as pool:
    #     for result in pool.imap_unordered(process_u_block, range(N_blocks)):
    #         h_in = h_in + result[0]
    #         h_out = h_out + result[1]
    

    # with mp.Pool(POOL_SIZE) as pool:
    #     results = [pool.apply_async(process_u_block, (i,)) for i in range(N_blocks)]

    with mp.Pool(POOL_SIZE) as pool:
        for result in pool.imap_unordered(process_u_block, range(N_blocks)):
            print('Completed block: ', result, flush=True)


if __name__ == '__main__':
    main()