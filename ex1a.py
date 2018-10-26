
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays

INPUT_PATH = 'data/face.mat'
TRAINING_SPLIT = 0.7


def import_processing(data):

    faces = loadmat(data)

    X = np.reshape(faces['X'], (46*56, 10, 52))

    X = split_data(X)

    means = [np.mean(x, axis=1) for x in X]
    data = [(x - means[i][..., None]) for i, x in enumerate(X)]
    return data, means


def split_data(X):

    training_data = np.reshape(X[..., 0:int(TRAINING_SPLIT*10), :], (46*56, -1))
    test_data = np.reshape(X[..., int(TRAINING_SPLIT*10):, :], (46*56, -1))
    data = [training_data, test_data]
    return data


def compute_S(data):

    N = data.shape[1] # Not needed
    S = np.cov(data, bias=True) # Normalises by N
    return S


def find_eigenvectors(S, how_many=-1):

    if how_many is -1:
        how_many = S.shape[0]
    eigvalues, eigvectors = np.linalg.eig(S)
    indices = np.flip(np.argsort(eigvalues), axis=0) # Gives original indices after sorting
    sorted_eigvalues = eigvalues[indices]
    sorted_eigvectors = eigvectors[:, indices]

    return sorted_eigvalues[0:how_many], sorted_eigvectors[:, 0:how_many]


def count_non_zero(eigenvalues):

    # boolean_mask = eigenvalues.nonzero()  # Mask of same shape as vector which is True if value is non zero
    boolean_mask = eigenvalues > 0.00001 # not really non-zero
    remaining_values = eigenvalues[boolean_mask]  # Only keep non-zero values
    return remaining_values.shape[0]  # How many left ?


def find_projection(eigenvectors, faces):  # eigenvectors and faces in vector form

    coeffs = np.matmul(faces.transpose(), eigenvectors)

    return coeffs


def reconstruct(eigenvectors, coeffs, mean):

    reconstructions = mean[:, None] + np.matmul(eigenvectors, coeffs.transpose())

    return reconstructions


def measure_reconstruct_error(originals, reconstructions):

    return np.linalg.norm(originals - reconstructions, ord=2, axis=0)

if __name__ == '__main__':
    X, means = import_processing(INPUT_PATH)
    S = compute_S(X[0])
    eig = find_eigenvectors(S, -1) # Compute all eigenvectors

    # eigenfaces = display_eigenvectors(eig[1])
    count = count_non_zero(eig[0])
    # save_image({'eigenfaces': eigenfaces})
    # save_dict = {'processedData': X[0], 'eigVal': eig[0], 'eigVec': eig[1], 'meanImage': means[0], 'nonZeroEig': count}
    # save_values(save_dict)

    print('Found {} non-zero eigenvalues'.format(count))
    error_vars = []
    error_means = []
    plt.figure(1)

    for i in range(1, count, 1):

        coeffs = find_projection(eig[1][:, :i], X[1])  # On test data
        reconstructed_image = reconstruct(eig[1][:, :i], coeffs, means[1])
        errors = measure_reconstruct_error(X[1] + means[1][:, None], reconstructed_image)

        error_var = np.var(errors)
        error_mean = np.mean(errors)
        error_vars.append(error_var)
        error_means.append(error_mean)
        # update plot
        plt.subplot(211)
        plt.plot(error_means)
        plt.title('Mean reconstruction error against number of eigenvectors')

        plt.subplot(212)
        plt.plot(error_vars)
        plt.title('Variance of reconstruction error on training images \nagainst number'
                  'of eigenvectors')
        plt.show(block=False)
        plt.pause(0.01)

    # To do: The error is in a random unit, may be make it percentage ? Same for variance

    save_dict = {'reconst_error_mean': error_means, 'reconst_error_var': error_vars}
    save_values(save_dict)