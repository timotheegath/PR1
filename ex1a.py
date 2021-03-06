
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays

INPUT_PATH = 'data/face.mat'
TRAINING_SPLIT_PERCENT = 0.7
TRAINING_SPLIT = int(TRAINING_SPLIT_PERCENT*10)
NUMBER_PEOPLE = 52



def import_processing(data, class_means=False):

    faces = loadmat(data)
    # faces dimension is 2576, 520 -> each image is column vector of pixels(46, 56)
    X = np.reshape(faces['X'], (46*56, 52, 10))  # separate arrays for each person
    X = split_data(X)
    if not class_means:
        means = [np.mean(X[0], axis=1)]
        data = [(x - means[0][..., None]) for i, x in enumerate(X)]
    else:
        x = X[0]
        means = [np.mean(x[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT], axis=1) for i in range(NUMBER_PEOPLE)]
        training_data = x
        for i in range(NUMBER_PEOPLE):
            training_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT] = x[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT]\
                                                                      - means[i][:, None]
        data = [training_data, X[1]]

    return data, means


def split_data(X):
    random_indexes = np.arange(0, 10)
    np.random.shuffle(random_indexes)

    training_data = np.reshape(X[..., random_indexes[0:TRAINING_SPLIT]], (46*56, -1))
    test_data = np.reshape(X[..., random_indexes[TRAINING_SPLIT:]], (46*56, -1))

    data = [training_data, test_data]
    return data


def compute_S(data, low_res=False):

    N = data.shape[1]
    if low_res:
        data = data.transpose()
    S = np.matmul(data, data.transpose()) / N # Normalises by N

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

    coeffs = np.matmul(faces.transpose(), eigenvectors).transpose()
    # number_of_eigenvectors X Faces
    return coeffs

def find_reference_coeffs(eigenvectors, faces):

    coeffs = find_projection(eigenvectors, faces)
    # There are 7 images per individual:
    #coeffs_sequence = np.split(coeffs, 7, axis=1)
    #stacked_sequence = np.stack(coeffs_sequence, axis=2)
    #coeffs_per_individual = np.mean(stacked_sequence, axis=2)
    # number of eigenvectors X number of individuals

    return coeffs




def accuracy_measurement(ground_truth, results):

    right = (ground_truth == results)
    accuracy = (right[right].shape[0])/(ground_truth.shape[0])

    return accuracy


def measure_reconstruct_error(originals, reconstructions):

    return np.linalg.norm(originals - reconstructions, ord=2, axis=0)


def regroup(ref_coeffs):

    clusters, labels, _= k_means(ref_coeffs.transpose(), n_clusters=NUMBER_PEOPLE, verbose=False)
    unique, index, counts = np.unique(labels, return_counts=True, return_index=True)
    return clusters.transpose()

def recognize(reference, to_classify, eigenvectors):

    coeffs_to_classify = np.real(find_projection(eigenvectors, to_classify))
    who_is_it = np.zeros(to_classify.shape[1], dtype=np.uint16)
    euclidean = True
    if euclidean:
        for i in range(to_classify.shape[1]):
            unknown = coeffs_to_classify[:, i][:, None]
            distance = unknown - reference
            distance = np.linalg.norm(distance, axis=0)
            who_is_it[i] = np.floor(np.argmin(distance)/TRAINING_SPLIT)

    else:

        coeffs_to_classify = normalize(coeffs_to_classify.transpose(), 'l2').transpose()
        reference = normalize(reference.transpose(), 'l2').transpose()
        dot_products = np.matmul(coeffs_to_classify.transpose(), reference)
        # number of faces to classify X number of reference individuals
        who_is_it = np.argmax(dot_products, axis=1)
        print(who_is_it.shape)
        # Returns the vector where each picture is assigned a number
    return who_is_it


if __name__ == '__main__':
    X, means = import_processing(INPUT_PATH)
    S = compute_S(X[0])
    eig = find_eigenvectors(S, -1) # Compute all eigenvectors

    # eigenfaces = display_eigenvectors(eig[1])
    count = count_non_zero(eig[0])
    # save_image({'eigenfaces': eigenfaces})
    save_dict = {'processedData': X[0], 'eigVal': eig[0], 'eigVec': eig[1], 'meanImage': means[0], 'nonZeroEig': count}
    save_values(save_dict)

    print('Found {} non-zero eigenvalues'.format(count))
    error_vars = []
    error_means = []
    plt.figure(1)
    #display_eigenvectors(X[1])
    for i in range(1, count, 1):

        # reconstructed_image = reconstruct(eig[1][:, :i], coeffs, means[1])
        # errors = measure_reconstruct_error(X[1] + means[1][:, None], reconstructed_image)
        ref_coeffs = np.real(find_reference_coeffs(eig[1][:, :i], X[0]))
        who_is_it = recognize(ref_coeffs, X[1], eig[1][:, :i])

        #who_is_it = recognize(ref_coeffs, X[1], eig[1][:, :i])
        true_individual_index = np.arange(0, NUMBER_PEOPLE)
        true_individual_index = np.repeat(true_individual_index[:, None], 10-TRAINING_SPLIT, axis=1).reshape(-1)
        accuracy = accuracy_measurement(true_individual_index, who_is_it)
        # error_var = np.var(errors)
        error_mean = np.mean(accuracy)

        # error_vars.append(error_var)
        error_means.append(error_mean)
        # update plot
        plt.subplot(211)
        plt.plot(error_means)
        plt.title('Mean reconstruction error against number of eigenvectors')

        # plt.subplot(212)
        # plt.plot(error_vars)
        # plt.title('Variance of reconstruction error on training images \nagainst number'
        #           'of eigenvectors')
        plt.show(block=False)
        plt.pause(0.01)

    # To do: The error is in a random unit, may be make it percentage ? Same for variance

    # save_dict = {'reconst_error_mean': error_means, 'reconst_error_var': error_vars}
    # save_values(save_dict)