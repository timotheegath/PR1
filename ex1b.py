import numpy as np
from ex1a import find_eigenvectors, compute_S
from in_out import load_arrays
from sklearn.preprocessing import normalize
from in_out import display_eigenvectors

previous_data = load_arrays('1a')
training_data = previous_data['processedData']  # training_data dimension is 2576, 364 -> each image is column vector of
                                                #  pixels(46, 56)
high_eigvals = previous_data['eigVal']
high_eigvecs = previous_data['eigVec']

matrix_AtA = np.matmul(training_data.transpose(), training_data)
eigvalues, eigvectors = np.linalg.eig(matrix_AtA)
low_eigvectors = np.matmul(training_data, eigvectors)  #low_eigvectors dimension is 2576, 364
indices = np.flip(np.argsort(eigvalues), axis=0) # Gives original indices after sorting
sorted_eigvalues = eigvalues[indices]
low_eigvectors = low_eigvectors[:, indices]
#norm = np.linalg.norm(low_eigvectors, axis=0, ord=2)[None, :]

low_eigvectors = low_eigvectors/np.linalg.norm(low_eigvectors, axis=0)[None, :]

high_eigvecs = high_eigvecs/np.linalg.norm(high_eigvecs, axis=0)[None, :]
print(np.linalg.norm(high_eigvecs, axis=0))
difference = np.matmul(high_eigvecs[..., :364 ], low_eigvectors.transpose())
print(difference)
