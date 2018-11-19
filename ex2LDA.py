import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means

from ex1a import count_non_zero
from in_out import display_eigenvectors, save_values


INPUT_PATH = 'data/face.mat'
TRAINING_SPLIT_PERCENT = 0.7
TRAINING_SPLIT = int(TRAINING_SPLIT_PERCENT*10)
NUMBER_PEOPLE = 52
M_PCA_reduction = 0  # Negative value
M_LDA_reduction = 0   # Negative value

# Leave those alone, access only
M_PCA = 0
M_LDA = 0
SB_RANK = 0
SW_RANK = 0

def import_processing(data, class_means=False):

    faces = loadmat(data)
    # faces dimension is 2576, 520 -> each image is column vector of pixels(46, 56)
    X = np.reshape(faces['X'], (46*56, 52, 10))  # separate arrays for each person
    X = split_data(X)
    means = np.mean(X[0], axis=1, keepdims=True)
    # data = [(x - means[0][..., None]) for i, x in enumerate(X)]
    return X, means


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


def reduce_by_PCA(training_data, means):
    global M_PCA

    training_data_norm = training_data - means
    low_S = compute_S(training_data_norm, low_res=True)
    eig_val, eig_vec = find_eigenvectors(low_S, how_many=-1)
    eig_vec = retrieve_low_eigvecs(eig_vec, training_data_norm)
    M_PCA = training_data_norm.shape[1]-NUMBER_PEOPLE + M_PCA_reduction   # hyperparameter Mpca <= N-c
    eig_vec_reduced = eig_vec[:, :M_PCA]
    return eig_vec_reduced


def compute_class_means(training_data):

    class_means = np.mean(training_data.reshape(-1, NUMBER_PEOPLE, TRAINING_SPLIT), axis=2) # Shape is 2576*52 -> D*c
    return class_means


def compute_class_scatters(training_data, class_means):

    class_means_expand = np.repeat(class_means, TRAINING_SPLIT, axis=1)
    class_means_expand = class_means_expand.reshape(-1, NUMBER_PEOPLE, TRAINING_SPLIT).transpose(1, 0, 2)
    training_data_resh = training_data.reshape(-1, NUMBER_PEOPLE, TRAINING_SPLIT).transpose(1, 0, 2)
    class_scatters = np.matmul(training_data_resh - class_means_expand, (training_data_resh - class_means_expand).transpose(0, 2, 1))
    # Might have to for loop but I think it works
    return class_scatters

def compute_Sb(class_means):

    global_mean = np.mean(class_means, axis=1, keepdims=True)
    global_mean = np.repeat(global_mean, NUMBER_PEOPLE, axis=1)
    Sb = np.matmul(class_means - global_mean, (class_means - global_mean).transpose())
    return Sb

def compute_Sw(class_scatters):

    Sw = np.sum(class_scatters, axis=0)
    return Sw


def compute_LDA_Fisherfaces(Sw, Sb, Wpca, faces):
    global M_LDA

    # Maybe remove mean from faces
    Sw_PCA = np.matmul(np.matmul(Wpca.transpose(), Sw), Wpca)
    Sb_PCA = np.matmul(np.matmul(Wpca.transpose(), Sb), Wpca)
    S = np.matmul(np.linalg.inv(Sw_PCA), Sb_PCA)
    eig_vals, fisherfaces = find_eigenvectors(S, how_many=-1)
    M_LDA = count_non_zero(eig_vals) + M_LDA_reduction     # hyperparameter Mlda <= c-1 -> there should be 51 non_zero eiganvalues
    # print(M_LDA)     # Mlda = c - 1 = 51
    fisherfaces_reduced = fisherfaces[:, :M_LDA]
    faces_PCA = find_projection(Wpca, faces)
    fisher_ref_coeffs = find_projection(fisherfaces_reduced, faces_PCA)
    return fisher_ref_coeffs, fisherfaces_reduced


def goto_original_domain(fisherfaces, Wpca):

    fisher_images = np.matmul(Wpca, fisherfaces)
    return fisher_images


def find_fisher_coeffs(candidate_images, Wpca, fisherfaces):

    PCA_images = find_projection(Wpca, candidate_images)
    LDA_coeffs = find_projection(fisherfaces, PCA_images) # 51 vector

    return LDA_coeffs


def classify(LDA_coeffs_training, LDA_coeffs_test):

    distances = []
    for i in range(LDA_coeffs_test.shape[1]):
        distances.append(np.linalg.norm(LDA_coeffs_training - LDA_coeffs_test[:, i][:, None], axis=0))

    return np.floor(np.argmin(np.array(distances), axis=1)/TRAINING_SPLIT).astype(np.uint16)


def create_ground_truth():

    true_individual_index = np.arange(0, NUMBER_PEOPLE)
    true_individual_index = np.repeat(true_individual_index[:, None], 10-TRAINING_SPLIT, axis=1).reshape(-1)
    return true_individual_index


def bool_and_accuracy(ground_truth, prediction):

    correct = ground_truth == prediction
    accuracy = (correct[correct].shape[0]) / (ground_truth.shape[0])
    return correct, accuracy


if __name__ == '__main__':

    M_PCAs = []
    accuracies = []

    while M_PCA_reduction > -312:

        [training_data, testing_data], means = import_processing(INPUT_PATH)
        Wpca = reduce_by_PCA(training_data, means)
        class_means = compute_class_means(training_data)
        class_scatters = compute_class_scatters(training_data, class_means)
        Sb = compute_Sb(class_means)
        SB_RANK =  np.linalg.matrix_rank(Sb)      # Rank is c - 1 -> 51
        # print(SB_RANK)
        Sw = compute_Sw(class_scatters)
        SW_RANK = np.linalg.matrix_rank(Sw)       # Rank is N - c -> 312(train_imgs) - 52 = 260 (same as PCA reduction)
        # print(SW_RANK)
        reference_LDA_coeffs, fisherfaces = compute_LDA_Fisherfaces(Sw, Sb, Wpca, training_data)
        # CHECKED THIS FAR

        # fish_images = goto_original_domain(fisherfaces, Wpca)
        # display_eigenvectors(fish_images)

        # ''' Start classification procedure'''
        candidate_LDA_coeffs = find_fisher_coeffs(testing_data, Wpca, fisherfaces)
        classification = classify(reference_LDA_coeffs, candidate_LDA_coeffs)

        ground_truth = create_ground_truth()

        bool_array, accuracy = bool_and_accuracy(ground_truth, classification)

        print(accuracy)

        accuracies.append(accuracy)
        M_PCAs.append(M_PCA)
        save_dict = {'accuracy': accuracies, 'training_split': TRAINING_SPLIT, 'M_PCA': M_PCAs, 'M_LDA': M_LDA,
                     'Sb_rank': SB_RANK, 'Sw_rank': SW_RANK}
        save_name = 'split_{}m_lda{}VARY_M_PCA'.format(TRAINING_SPLIT, M_LDA)
        save_values(save_dict, name=save_name)

        M_PCA_reduction -= 15
