
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat

INPUT_PATH = 'data/face.mat'
TRAINING_SPLIT = 0.7

def import_processing(data):

    faces = loadmat(data)
    X = np.reshape(faces['X'], (46, 56, 10, 52))
    X = X.transpose()

    X = split_data(X)
    means = [np.mean(x, axis=2) for x in X]
    data = [x - means[i][..., None] for i, x in enumerate(X)]
    return data


def split_data(X):

    training_data = np.reshape(X[..., 0:int(TRAINING_SPLIT*10), :], (46, 56, -1))
    test_data = np.reshape(X[..., int(TRAINING_SPLIT*10):, :], (46, 56, -1))
    data = [training_data, test_data]

    return data


X = import_processing(INPUT_PATH)
