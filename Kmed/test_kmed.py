from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

import Kmed

data = np.array([[1,-10,4],
                 [2,5,2],
                 [10,3,7],
                 [6,-2,1],
                 [7,9,0],
                 [12,1,-1]])

# distance matrix
D = pairwise_distances(data, metric='euclidean')

# split into 3 clusters
M, C = Kmed.kMedoids(D, 3)

print(D)
print('medoids:')
for point_idx in M:
    print( data[point_idx] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, data[point_idx]))