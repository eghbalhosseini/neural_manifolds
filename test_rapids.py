import cudf
import numpy as np
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors


n_samples = 2**17
n_features = 40

n_query = 2**13
n_neighbors = 4
random_state = 0


print(f'n_samples: {n_samples}')
print('done!')