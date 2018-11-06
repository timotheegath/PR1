import numpy as np
from ex1a import find_eigenvectors, compute_S, import_processing, INPUT_PATH
import matplotlib.pyplot as plt
from in_out import load_arrays
from sklearn.preprocessing import normalize
from in_out import display_eigenvectors

USE_PREVIOUS  = False

def retrieve_low_eigvecs(low_eigvecs, data): # Returns normalized eigenvectors

    vecs = np.matmul(data, low_eigvecs)
    vecs /= np.linalg.norm(vecs, axis=0)[None, :]
    return vecs


if USE_PREVIOUS:
    previous_data = load_arrays('1a')
    training_data = previous_data['processedData']  # training_data dimension is 2576, 364 -> each image is column vector of
                                                    #  pixels(46, 56)
    
    high_eigvals = previous_data['eigVal']
    high_eigvecs = previous_data['eigVec']
    
else :
    [training_data, _], means = import_processing(INPUT_PATH)
    matrix_AtA = np.matmul(training_data.transpose(), training_data)
    matrix_AtA /= training_data.shape[1]
    low_eigvalues, fake_low_eigvecs = find_eigenvectors(matrix_AtA, -1)  # Compute all eigenvectors
    low_eigvecs = retrieve_low_eigvecs(fake_low_eigvecs, training_data)  #low_eigvecs dimension is 2576, 364

    # Recompute

    high_eigvals, high_eigvecs = find_eigenvectors(compute_S(training_data), -1)



difference = np.matmul(high_eigvecs[..., :364 ], low_eigvecs.transpose())
print(difference)
eigenvalue_difference = high_eigvals[:364] - low_eigvalues
print(np.min(eigenvalue_difference), np.max(eigenvalue_difference), np.mean(eigenvalue_difference))
plt.figure(1)
plt.scatter(np.arange(0, low_eigvalues.shape[0]), low_eigvalues, c='b', marker='o')
plt.scatter(np.arange(0, low_eigvalues.shape[0]), high_eigvals[:364], c='r', marker='x')
plt.show()