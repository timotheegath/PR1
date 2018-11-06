import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays
from ex1a import find_eigenvectors, find_projection, import_processing, INPUT_PATH, compute_S
from ex1b import retrieve_low_eigvecs
from in_out import display_eigenvectors
import time

def reconstruct(eigenvectors, coeffs, mean):

    reconstructions = mean[:, None] + np.matmul(eigenvectors, coeffs)

    return reconstructions


def recognize(reference, to_classify, eigenvectors):

    coeffs_to_classify = np.real(find_projection(eigenvectors, to_classify))
    who_is_it = np.zeros(to_classify.shape[1], dtype=np.uint16)
    euclidean = True
    if euclidean:
        for i in range(to_classify.shape[1]):
            unknown = coeffs_to_classify[:, i][:, None]
            distance = unknown - reference
            distance = np.linalg.norm(distance, axis=0)
            who_is_it[i] = np.floor(np.argmin(distance)/7)

    else:

        coeffs_to_classify = normalize(coeffs_to_classify.transpose(), 'l2').transpose()
        reference = normalize(reference.transpose(), 'l2').transpose()
        dot_products = np.matmul(coeffs_to_classify.transpose(), reference)
        # number of faces to classify X number of reference individuals
        who_is_it = np.argmax(dot_products, axis=1)
        print(who_is_it.shape)
        # Returns the vector where each picture is assigned a number
    return who_is_it

def measure_reconstruction_error(reconstructed, original):

    difference = np.linalg.norm(reconstructed - original, axis=0) ** 2
    distortion = np.mean(difference)
    return distortion


[training_data, test_data], means = import_processing(INPUT_PATH)
eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), -1)
eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
projections = find_projection(eigenvectors, training_data)
distortions = []
for i in range(1, eigenvalues.shape[0], 1):
    try :
        temp_projections = projections[:i]
        temp_eigenvecs = eigenvectors[:, :i]
        reconstructions = reconstruct(temp_eigenvecs, temp_projections, means[0])
        distortion = measure_reconstruction_error(reconstructions, training_data + means[0][..., None])
        distortions.append(distortion)
        display_eigenvectors(reconstructions[:, :30])

        plt.plot(distortions)
        plt.title('Distortion against number of eigenvectors')
        plt.show(block=False)
        plt.pause(0.01)
    except KeyboardInterrupt:
        continue


time.sleep(20)

