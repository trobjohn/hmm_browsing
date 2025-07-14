import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale


def min_max(var):
    var = (var-np.min(var))/(np.max(var)-np.min(var))
    return var

## Run clustering algorithm on the page embedding
rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
page_embed = pickle.load(open('./sweep/page_embed.sav', 'rb'))


import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

def min_max(var):
    var = (var-np.min(var))/(np.max(var)-np.min(var))
    return var

## Run clustering algorithm on the page embedding
rolo_s_clean = pickle.load(open('./sweep/rolo_s_clean.sav','rb')) 
X = pickle.load(open('./sweep/page_embed.sav', 'rb'))
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

#H = pickle.load(open('./models/H.sav', 'rb'))

print(X.shape)

print('Running clustering algorithm')

# from sklearn.cluster import OPTICS, cluster_optics_dbscan
# optics = OPTICS()#min_samples=50, xi=0.05, min_cluster_size=0.05)
# optics.fit(page_embed)
# pickle.dump(optics, open('./models/optics.sav', 'wb'))
# gdf = pd.DataFrame({'domain':rolo_s_clean, 'cluster':clust.labels_[clust.ordering_]})
# gdf.to_csv('./data/optics_clusters.csv')

# K means clustering scheme
from sklearn.cluster import KMeans
kmc = KMeans( n_clusters = 500, max_iter=1000, random_state=0, n_init='auto')
kmc = kmc.fit(X)
pickle.dump(kmc, open('./sweep/kmc.sav', 'wb'))
gdf = pd.DataFrame({'domain':rolo_s_clean, 'cluster':kmc.labels_})
gdf.to_csv('./sweep/kmc_clusters.csv')
# kmc.labels_
# gdf = pd.DataFrame({'labs':kmc.labels_, 'x':page_embed[:,0], 'y':page_embed[:,1]})
# sns.scatterplot(x='x',y='y' , hue = 'labs', data = gdf)

# # Agglomerative Clustering Scheme
# from sklearn.cluster import AgglomerativeClustering
# ac = AgglomerativeClustering( n_clusters = 100)
# ac.fit(page_embed)
# pickle.dump(ac, open('./models/ac.sav', 'wb'))
# # ac.labels_
# # gdf = pd.DataFrame({'labs':ac.labels_, 'x':page_embed[:,0], 'y':page_embed[:,1]})
# # sns.scatterplot(x='x',y='y' , hue = 'labs', data = gdf)
# gdf = pd.DataFrame({'domain':rolo_s_clean, 'cluster':ac.labels_})
# gdf.to_csv('./data/ac_clusters.csv')
