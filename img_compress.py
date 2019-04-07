# -*- coding: utf-8 -*-

from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('lena.png')

rows = image.shape[0]
cols = image.shape[1]
print(rows)
print(cols)
image = image.reshape(image.shape[0]*image.shape[1],3)
kmeans = KMeans(n_clusters = 8, n_init=10, max_iter=200)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels = labels.reshape(rows,cols)

np.save('codebook_lena.npy',clusters)    
io.imsave('compressed_lena0.png',labels)