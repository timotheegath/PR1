
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from in_out import display_eigenvectors, display_single_image, save_image, save_values

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

    N = data.shape[1]
    S = np.cov(data, bias=True)
    return S


def find_eigenvectors(S, how_many=-1):

    if how_many is -1:
        how_many = S.shape[0]
    eigvalues, eigvectors = np.linalg.eig(S)
    indices = np.flip(np.argsort(eigvalues), axis=0) # Gives original indices after sorting
    sorted_eigvalues = eigvalues[indices]
    sorted_eigvectors = eigvectors[:, indices]

    return sorted_eigvalues[0:how_many], sorted_eigvectors[:, 0:how_many]


save_values({'arrayA': np.zeros((5,6)), 'ArrayB': np.zeros((6,6))})
X, means = import_processing(INPUT_PATH)
# On training data
S = compute_S(X[0])
eig = find_eigenvectors(S, 30)
display_eigenvectors(eig[1])
