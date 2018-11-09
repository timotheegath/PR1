import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays
from ex1a import find_eigenvectors, find_projection, import_processing, INPUT_PATH, compute_S, recognize, NUMBER_PEOPLE, TRAINING_SPLIT, count_non_zero
from ex1b import retrieve_low_eigvecs
from ex1aa import reconstruct
from in_out import display_eigenvectors
import time
import cv2


NUMBER_OF_EIGENVECTORS = -1


def identify_failure(bool_a, number=10):

    indices = np.argsort(bool_a)  # Gives original indices after sorting
    return indices[:number]


def identify_success(bool_a, number=10):

    indices = np.flip(np.argsort(bool_a), axis=0)  # Gives original indices after sorting
    return indices[:number]

def confusion_matrix(ground_truth, prediction, res=80):

    res = NUMBER_PEOPLE * res
    matrix = np.zeros((ground_truth.shape[0], NUMBER_PEOPLE, NUMBER_PEOPLE), dtype=np.float32)
    big_matrix = np.zeros((res, res), dtype=np.float32)
    small_index = np.floor(np.linspace(0, NUMBER_PEOPLE**2, res**2, endpoint=False)).astype(np.uint16)
    big_index = np.arange(0, res**2)
    for i in range(ground_truth.shape[0]):
        matrix[i, ground_truth[i], prediction[i]] = 1
    matrix = np.sum(matrix, axis=0)
    matrix /= 3
    print(np.max(small_index), np.max(big_index))
    big_matrix.flatten()[big_index] = matrix.flatten()[small_index]

    # matrix = cv2.resize(matrix, dsize=(res,res), interpolation=cv2.INTER_LINEAR)

    return 1 - big_matrix.reshape((res, res))

def bool_and_accuracy(ground_truth, prediction):

    correct = ground_truth == prediction
    accuracy = (correct[correct].shape[0]) / (ground_truth.shape[0])
    return correct, accuracy


def create_ground_truth():

    true_individual_index = np.arange(0, NUMBER_PEOPLE)
    true_individual_index = np.repeat(true_individual_index[:, None], 3, axis=1).reshape(-1)
    return true_individual_index



def classify(projections_training, projections_test):

    distances = []
    for i in range(projections_test.shape[1]):
        distances.append(np.linalg.norm(projections_training - projections_test[:, i][:, None], axis=0))
    return np.floor(np.argmin(np.array(distances), axis=1)/7).astype(np.uint16)

def classify_Rec(query_images, eigenvectors, means):

    errors = np.zeros((query_images.shape[1], NUMBER_PEOPLE))
    for i, vec in enumerate(eigenvectors):

        projection = np.matmul((query_images-means[i][:, None]).transpose(), vec)
        reconstruction = reconstruct(vec, projection.transpose(), means[i])
        error = np.linalg.norm((reconstruction-query_images), axis=0)  # Mean per class or not ?
        errors[:, i] = error

    classification = np.argmin(errors, axis=1)
    return classification







if __name__ == '__main__':
    NN = False
    t1 = time.time()

    if NN:
        [training_data, testing_data], means = import_processing(INPUT_PATH)
        eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), -1)
        eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
        projections_training, projections_test = find_projection(eigenvectors, training_data),\
                                                 find_projection(eigenvectors, testing_data)
        recognised_faces = classify(projections_training, projections_test)
    
        true_faces = create_ground_truth()
    
        bool_recognised, accuracy = bool_and_accuracy(true_faces, recognised_faces)
        conf_matrix = confusion_matrix(true_faces, recognised_faces, res=20)*255
    
        # print(accuracy)
    
    
        cv2.imshow('Confusion matrix', conf_matrix)
        cv2.waitKey()
        print(np.unique(conf_matrix))
    else:
        [training_data, testing_data], means = import_processing(INPUT_PATH, class_means=True)
        eigenvectors = []
        for i in range(NUMBER_PEOPLE):
            eigv, eigvec = find_eigenvectors(compute_S(training_data[:, i*7:(i+1)*7], low_res=True), -1)
            eigvec = retrieve_low_eigvecs(eigvec, training_data[:, i*7:(i+1)*7])
            no_non_zero = count_non_zero(eigv)
            eigvec = eigvec[:, :no_non_zero]
            eigenvectors.append(eigvec)
        classifications = classify_Rec(testing_data, eigenvectors, means)
        true_faces = create_ground_truth()
        bool_recognised, accuracy = bool_and_accuracy(true_faces, classifications)
        print(accuracy)
    t2 = time.time()
    duration = t2-t1

    print(duration)
    
    # print(recognised_faces)
    # print(bool_recognised)
