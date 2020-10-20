# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:18:02 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

wine=pd.read_csv('C:/Users/HP/Desktop/assignments submission/PCA/wine.csv')
wine.isna().sum()
#normalize the data
from sklearn.preprocessing import scale
norm=scale(wine.iloc[:,1:])
norm_data=pd.DataFrame(norm)

############K-Means clustering with out PCA##########

k=list(range(2,15))
k
TWSS=[]
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(norm_data)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(norm_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_data.shape[1]),metric='euclidean')))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-');plt.xlabel('k value');plt.ylabel('TWSS')
#k=4
model_k=KMeans(n_clusters=4).fit(norm_data)
clusters_k_means=pd.DataFrame(model_k.labels_)
k_means=clusters_k_means.value_counts()

############Hierarchial Clustering without PCA#################

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(norm_data)
h_clusters=pd.DataFrame(h_complete.labels_)
hierarchy=h_clusters.value_counts()


#################################################################

############################### PCA ############################

from sklearn.decomposition import PCA
pca=PCA()
pca_values=pca.fit_transform(norm_data)
pca_values.shape
var=pca.explained_variance_ratio_
var
cum_var=np.cumsum(np.round(var,decimals=4)*100)
cum_var

###############K-Means with PCA##############

#PCA data with 3 Principal component scores
pca_data=pd.DataFrame(pca_values)
pca_data=pca_data.drop(columns=[3,4,5,6,7,8,9,10,11,12])
k1=list(range(2,15))
k1
TWSS1=[]
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data)
    WSS1=[]
    for j in range(i):
        WSS1.append(sum(cdist(pca_data.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,pca_data.shape[1]),metric='euclidean')))
    TWSS1.append(sum(WSS1))

plt.plot(k1,TWSS1,'bo-');plt.xlabel('k1 value');plt.ylabel('TWSS1')
#k=4,with PC1,PC2,PC3(3 columns) also we got same k value as for with original data(13 columns)
model1=KMeans(n_clusters=4).fit(pca_data)
clusters_pca_k=pd.DataFrame(model1.labels_)
k_means_pca=clusters_pca_k.value_counts()

#########Hierarchial with PCA#############

from sklearn.cluster import AgglomerativeClustering
h_pca_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(pca_data)
h_pca_clusters=pd.DataFrame(h_pca_complete.labels_)
hierarchy_pca=h_pca_clusters.value_counts()

#comparision of clusters 
comparision=pd.concat([k_means,hierarchy,k_means_pca,hierarchy_pca],axis=1)
comparision
comparision.rename(columns={0:'k-means',1:'hierarchy',2:'k-means with PCA',3:'hierarchy with PCA'},inplace=True)
