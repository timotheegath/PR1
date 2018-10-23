
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat


def import_processing(data):

    faces = loadmat(data)
    X = np.reshape(faces['X'], (46, 56, 520))
    X = X.transpose
    return X

def split_data(X)



X = import_processing('data/face.mat')