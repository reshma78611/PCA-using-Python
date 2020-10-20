# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:51:09 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

univ=pd.read_csv('C:/Users/HP/Desktop/datasets/Universities.csv')
univ.isna().sum()

from sklearn.preprocessing import scale
norm_data=scale(univ.iloc[:,1:])
norm_data

#############PCA############
from sklearn.decomposition import PCA
pca=PCA()
pca_values=pca.fit_transform(norm_data)
pca_values.shape
#amount of variance of each PCA
var=pca.explained_variance_ratio_
var
#cumulative variance
cum_var=np.cumsum(np.round(var,decimals=4)*100)
cum_var
#variance plot for PCA components
plt.plot(cum_var,'r')
#plot between PCA1 and PCA2
x=pca_values[:,0]
y=pca_values[:,1]
plt.plot(x,y,'ro');plt.xlabel('PCA1');plt.ylabel('PCA2')
# no where pca1 and pca2 are correlated
plt.plot(np.arange(25),x,"ro")
