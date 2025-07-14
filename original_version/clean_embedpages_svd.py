import os as os
import numpy as np
from scipy.sparse import lil_matrix, hstack
from scipy.sparse.linalg import svds
import pickle
from sklearn.decomposition import PCA

if __name__ == '__main__':

    ## Convert to column sparse:
    print('Creating embedding...')  
    H_in = pickle.load(open('./models/H_in.sav', 'rb'))
    H_in = H_in.astype(float)
    H_out = pickle.load(open('./models/H_out.sav', 'rb'))
    H_out = H_out.astype(float)
    H = hstack( (H_in , H_out ) )
    del H_in, H_out

    ### Sparse SVD:
    # page_embed,sigma,v = svds(A = H, k = 100) # 100 was too much; what is the R2 of the embedding?
    # pickle.dump(sigma, open('./sweep/sigma.sav', 'wb')) # save eigenvalues
    # pickle.dump(page_embed, open('./sweep/page_embed.sav', 'wb')) # save page embedding

    ## PCA:
    n_comp = 10
    pca = PCA(n_components = n_comp, svd_solver='arpack')
    page_embed = pca.fit_transform(H)

    # sns.lineplot(x=np.arange(1,n_comp+1),y=pca.explained_variance_ratio_)
    # plt.show()
    # sns.lineplot(x=np.arange(1,n_comp+1),y=np.cumsum(pca.explained_variance_ratio_))
    # plt.show()
    
    print('Variance explained: ', np.sum(pca.explained_variance_ratio_) )
    pickle.dump(page_embed, open('./sweep/page_embed.sav', 'wb')) # save page embedding


