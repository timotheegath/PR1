import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from ex1a import count_non_zero


INPUT_PATH = 'data/face.mat'
TRAINING_SPLIT_PERCENT = 0.6
TRAINING_SPLIT = int(TRAINING_SPLIT_PERCENT*10)
NUMBER_PEOPLE = 52


def import_processing(data, class_means=False):

    faces = loadmat(data)
    # faces dimension is 2576, 520 -> each image is column vector of pixels(46, 56)
    X = np.reshape(faces['X'], (46*56, 52, 10))  # separate arrays for each person
    X = split_data(X)
    means = [np.mean(X[0], axis=1)]
    data = [(x - means[0][..., None]) for i, x in enumerate(X)]
    return data, means


def split_data(x):
    random_indexes = np.arange(0, 10)
    np.random.shuffle(random_indexes)

    training_data = np.reshape(x[..., random_indexes[0:TRAINING_SPLIT]], (46*56, -1))
    test_data = np.reshape(x[..., random_indexes[TRAINING_SPLIT:]], (46*56, -1))

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

def retrieve_low_eigvecs(low_eigvecs, data): # Returns normalized eigenvectors

    vecs = np.matmul(data, low_eigvecs)
    vecs /= np.linalg.norm(vecs, axis=0)[None, :]
    return vecs


def find_projection(eigenvectors, faces):  # eigenvectors and faces in vector form

    coeffs = np.matmul(faces.transpose(), eigenvectors).transpose()
    # number_of_eigenvectors X Faces
    return coeffs


def reduce_by_PCA(training_data):

    low_S = compute_S(training_data, low_res=True)
    eig_val, eig_vec = find_eigenvectors(low_S, how_many=-1)
    eig_vec = retrieve_low_eigvecs(eig_vec, training_data)
    Mpca = training_data.shape[1]-NUMBER_PEOPLE     # hyperparameter Mpca <= N-c
    eig_vec_reduced = eig_vec[:, :Mpca]
    projection_coeffs = find_projection(eig_vec_reduced, training_data)
    # return eig_vec_reduced
    return projection_coeffs


def compute_class_means(training_data):

    class_means = np.mean(training_data.reshape(-1, TRAINING_SPLIT, NUMBER_PEOPLE), axis=1) # Shape is 2576*52 -> D*c
    return class_means


def compute_class_scatters(training_data, class_means):

    class_means = np.repeat(class_means, TRAINING_SPLIT, axis=1)
    class_means = class_means.reshape(-1, TRAINING_SPLIT, NUMBER_PEOPLE).transpose(2, 0, 1)
    training_data = training_data.reshape(-1, TRAINING_SPLIT, NUMBER_PEOPLE).transpose(2, 0, 1)
    # print(training_data.shape, class_means.shape)
    class_scatters = np.matmul(training_data - class_means, (training_data - class_means).transpose(0, 2, 1))
    # Might have to for loop but I think it works
    # print(class_scatters.shape)
    return class_scatters

def compute_Sb(class_means):

    global_mean = np.mean(class_means, axis=1)
    global_mean = np.repeat(global_mean[:, None], NUMBER_PEOPLE, axis=1)
    Sb = np.matmul(class_means - global_mean, (class_means - global_mean).transpose())
    # print(Sb.shape)
    return Sb

def compute_Sw(class_scatters):

    Sw = np.sum(class_scatters, axis=0)
    # print(Sw.shape)
    return Sw


def compute_LDA_Fisherfaces(Sw, Sb):

    S = np.matmul(np.linalg.inv(Sw), Sb)
    eig_vals, fisherfaces = find_eigenvectors(S, how_many=-1)
    Mlda = count_non_zero(eig_vals)     # hyperparameter Mlda <= c-1 -> there should be 51 non_zero eiganvalues
    # print(Mlda)     # Mlda = c - 1 = 51
    fisherfaces_reduced = fisherfaces[:, :Mlda]
    return fisherfaces_reduced

if __name__ == '__main__':

    [training_data, testing_data], means = import_processing(INPUT_PATH)    # Training and Testing data have the
    # training mean removed
    reduced_training_data = reduce_by_PCA(training_data)
    # Wpca = reduce_by_PCA(training_data)
    # print(Wpca.shape)
    class_means = compute_class_means(reduced_training_data)
    class_scatters = compute_class_scatters(reduced_training_data, class_means)
    Sb = compute_Sb(class_means)
    print(Sb.shape, np.linalg.matrix_rank(Sb))        # Rank is c - 1 -> 51
    Sw = compute_Sw(class_scatters)
    print(Sw.shape, np.linalg.matrix_rank(Sw))        # Rank is N - c -> 312(train_imgs) - 52 = 260 (same as PCA reduction
    # projection)
    fisherfaces = compute_LDA_Fisherfaces(Sw, Sb)
    print(fisherfaces.shape)
