import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays
from ex1a import find_eigenvectors, find_projection, import_processing, INPUT_PATH, compute_S, recognize, NUMBER_PEOPLE, TRAINING_SPLIT
from ex1b import retrieve_low_eigvecs
from in_out import display_eigenvectors
import time

NUMBER_OF_EIGENVECTORS = -1


def identify_failure(bool_a, number=10):

    indices = np.argsort(bool_a)  # Gives original indices after sorting
    return indices[:number]


def identify_success(bool_a, number=10):

    indices = np.flip(np.argsort(bool_a), axis=0)  # Gives original indices after sorting
    return indices[:number]

def confusion_matrix(ground_truth, prediction, res=(80, 80)):

    res = res * NUMBER_PEOPLE * (1-TRAINING_SPLIT)*10
    matrix = np.zeros((res, res), dtype=np.float32)
    matrix[ground_truth, prediction] += 1
    matrix /= 3
    return 1 - matrix

def bool_and_accuracy(ground_truth, prediction):

    correct = ground_truth == prediction
    accuracy = (correct[correct].shape[0]) / (ground_truth.shape[0])
    return correct, accuracy


def create_ground_truth():

    true_individual_index = np.arange(0, NUMBER_PEOPLE)
    true_individual_index = np.repeat(true_individual_index[:, None], 3, axis=1).reshape(-1)
    return true_individual_index




if __name__ == '__main__':

    [training_data, testing_data], means = import_processing(INPUT_PATH)
    eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), NUMBER_OF_EIGENVECTORS)
    eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
    projections_training, projections_test = find_projection(eigenvectors, training_data),\
                                             find_projection(eigenvectors, testing_data)
