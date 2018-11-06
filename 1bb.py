import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays
from ex1a import find_eigenvectors, find_projection, import_processing, INPUT_PATH, compute_S, recognize
from ex1b import retrieve_low_eigvecs
from in_out import display_eigenvectors
import time


def classify(projections_training, projections_test):

    distances = []
    for i in range(projections_test.shape[1]):
        distances.append(np.linalg.norm(projections_training - projections_test[:, i][:, None], axis=0))
    return np.floor(np.argmin(np.array(distances), axis=1)/7).astype(np.uint16)


if __name__ == '__main__':

    [training_data, testing_data], means = import_processing(INPUT_PATH)
    eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), -1)
    eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
    projections_training, projections_test = find_projection(eigenvectors, training_data),\
                                             find_projection(eigenvectors, testing_data)
    recognised_faces = classify(projections_training, projections_test)
    print(recognised_faces)
    