import numpy as np
from ex1a import find_eigenvectors, compute_S, import_processing, INPUT_PATH
import matplotlib.pyplot as plt
from in_out import load_arrays
import time
from sklearn.preprocessing import normalize
from in_out import display_eigenvectors

USE_PREVIOUS  = False

def retrieve_low_eigvecs(low_eigvecs, data): # Returns normalized eigenvectors

    vecs = np.matmul(data, low_eigvecs)
    vecs /= np.linalg.norm(vecs, axis=0)[None, :]
    return vecs


if __name__ == '__main__':

    if USE_PREVIOUS:
        previous_data = load_arrays('1a')
        training_data = previous_data['processedData']  # training_data dimension is 2576, 364 -> each image is column vector of
                                                        #  pixels(46, 56)

        high_eigvals = previous_data['eigVal']
        high_eigvecs = previous_data['eigVec']

    else :
        [training_data, _], means = import_processing(INPUT_PATH)
        t0 = time.time()
        matrix_AtA = compute_S(training_data, low_res=True)
        t1 = time.time()
        low_eigvalues, fake_low_eigvecs = find_eigenvectors(matrix_AtA, -1)  # Compute all eigenvectors
        low_eigvecs = retrieve_low_eigvecs(fake_low_eigvecs, training_data)  #low_eigvecs dimension is 2576, 364
        tl = time.time()
        # Recompute

        high_eigvals, high_eigvecs = find_eigenvectors(compute_S(training_data), -1)
        th = time.time()
        print('AtA only : {} s ; Low-res full: {} s; High-res full: {} s'.format(t1-t0, tl-t0, th-tl))


    difference = np.matmul(high_eigvecs[..., :364 ].transpose(), low_eigvecs)

    eigenvalue_difference = high_eigvals[:364] - low_eigvalues

    plt.figure(1)
    plt.subplot(311)
    plt.scatter(np.arange(0, low_eigvalues.shape[0]), low_eigvalues, c='b', marker='o')
    plt.scatter(np.arange(0, low_eigvalues.shape[0]), high_eigvals[:364], c='r', marker='x')
    plt.legend(['Low-dimensional computation', 'High-dimensional computation'])
    plt.title('Eigenvalues')
    plt.subplot(312)
    plt.title('Difference between the high-res and low-res eigenvalues')
    plt.scatter(np.arange(0, low_eigvalues.shape[0]), eigenvalue_difference, marker='+')

    diag = np.eye(low_eigvalues.shape[0]).astype(np.bool_)
    plt.subplot(313)
    plt.title('Dot product of normalized eigenvectors obtained through the two computations')
    plt.scatter(np.arange(0, low_eigvalues.shape[0]), difference[diag], marker='+')
    plt.show()