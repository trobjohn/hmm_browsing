## Succeeded


import os as os
import pandas as pd
#import polars as pl
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import seaborn as sns
import pickle
import multiprocessing as mp
from multiprocessing import set_start_method
import random

# def task(i):    
#     print('Processing assignment: ', i, flush=True)    
#     result_path = '/scratch/trj2j/hmm/h/h_'
#     h_in_i = pickle.load(open(result_path+'in_'+str(i)+'.pickle', 'rb'))
#     h_out_i = pickle.load(open(result_path+'out_'+str(i)+'.pickle', 'rb'))    
#     return h_in_i, h_out_i

def main():

    POOL_SIZE = 12  # Rivanna cores, about 25 cores, 50 gb ram
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
    K = len(focus) # Number of focus sites
    J = len(rolo_s) # Total number of sites
    pickle.dump(rolo_s, open('./sweep/rolo_s_clean.pickle', 'wb'))
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

    ################################################################################################
    ################################################################################################

    h_in = lil_matrix((J,K), dtype=int) # other, focus
    #h_out = lil_matrix((J,K), dtype=int)

    result_path = '/scratch/trj2j/hmm/h/h_'
    for i in range(N_blocks):
        h_in_i = pickle.load(open(result_path+'in_'+str(i)+'.pickle', 'rb'))
        #h_out_i = pickle.load(open(result_path+'out_'+str(i)+'.pickle', 'rb'))    
        h_in += h_in_i
        #h_out += h_out_i
        print(' Pct cmp: ', i/N_blocks)

    # set_start_method("spawn")
    # with mp.Pool(POOL_SIZE) as pool:
    #     for result in pool.imap_unordered(task, range(N_blocks)):
    #         h_in += result[0]
    #         h_out += result[1]

    # with mp.Pool(POOL_SIZE) as pool:
    #     result = pool.map(task, range(N_blocks))
    #     for out in result.get():
    #         h_in += out[0]
    #         h_out += out[1]

    # # ## Convert to column sparse and run SVD
    print('Creating embedding...')  
    H_in = h_in.tocsr()
    # H_out = h_out.tocsr()
    del h_in#, h_out
    # H = hstack( (H_in,H_out) )
    pickle.dump(H_in, open('./sweep/H_in.sav', 'wb')) # save sparse matrices?
    # del H_in, H_out
    # page_embed,sigma,v = svds(A = H, k = 60) # 100 was too much
    # pickle.dump(sigma, open('./models/sigma.sav', 'wb')) # save eigenvalues
    # pickle.dump(page_embed, open('./models/page_embed.sav', 'wb')) # save page embedding
    # del sigma, v

################################################################################################
################################################################################################
################################################################################################

if __name__ == '__main__':
    main()