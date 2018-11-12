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
    means = [np.mean(X[0], axis=1)]
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


def find_projection(eigenvectors, faces):  # eigenvectors and faces in vector form

    coeffs = np.matmul(faces.transpose(), eigenvectors).transpose()
    # number_of_eigenvectors X Faces
    return coeffs





def compute_class_means(training_bag):
    class_means = np.zeros((training_bag.get_dim(), NUMBER_PEOPLE))

    for c, data in training_bag:
        class_means[:, c] = np.mean(data, axis=1) # Shape is 2576*number_of_sample_of_class -> D*c

    return class_means


def compute_class_scatters(training_data, class_means):

    class_means = np.repeat(class_means, TRAINING_SPLIT, axis=1)
    class_means = class_means.reshape(-1, TRAINING_SPLIT, NUMBER_PEOPLE).transpose(2, 0, 1)
    training_data = training_data.reshape(-1, TRAINING_SPLIT, NUMBER_PEOPLE).transpose(2, 0, 1)
    print(training_data.shape, class_means.shape)
    print(training_data.nbytes)
    meaned_training_data = training_data - class_means
    # Memory error if it's a 1.3 Gb matrix (52 * 2576 * 2576 in float64 !!!)
    class_scatters = np.zeros((training_data.shape[0], training_data.shape[1], training_data.shape[1]), dtype=np.float32)
    np.matmul(meaned_training_data, meaned_training_data.transpose(0, 2, 1), class_scatters)
    # Might have to for loop but I think it works
    return class_scatters

def compute_Sb(class_means):

    global_mean = np.mean(class_means, axis=1, keepdims=True)
    global_mean = np.repeat(global_mean, NUMBER_PEOPLE, axis=1)
    Sb = np.matmul(class_means - global_mean, (class_means - global_mean).transpose())
    # print(Sb.shape)
    return Sb

def compute_Sw(class_scatters):

    Sw = np.sum(class_scatters, axis=0)
    # print(Sw.shape)
    return Sw


class PCA_unit():

    def __init__(self):
        self.Wpca = 0
        self.ref_coeffs = [] * NUMBER_PEOPLE



    def train(self, training_bag):
        def retrieve_low_eigvecs(low_eigvecs, data):  # Returns normalized eigenvectors

            vecs = np.matmul(data, low_eigvecs)
            vecs /= np.linalg.norm(vecs, axis=0)[None, :]
            return vecs
        global M_PCA
        training_data = training_bag.get_all()

        training_data_meaned = training_data - np.mean(training_data, axis=1, keepdims=True)
        low_S = compute_S(training_data_meaned, low_res=True)
        eig_val, eig_vec = find_eigenvectors(low_S, how_many=-1)
        eig_vec = retrieve_low_eigvecs(eig_vec, training_data_meaned)
        # print(eig_vec.shape)
        M_PCA = training_data.shape[1]-NUMBER_PEOPLE + M_PCA_reduction   # hyperparameter Mpca <= N-c
        eig_vec_reduced = eig_vec[:, :M_PCA]

        self.Wpca = eig_vec_reduced
        self.ref_coeffs = [[]] * NUMBER_PEOPLE

        for c, data in training_bag:

            self.ref_coeffs[c] = self.__call__(data)

    def __call__(self, faces):

        coeffs = np.matmul(faces.transpose(), self.Wpca).transpose()
        return coeffs


class LDA_unit():

    def __init__(self):

        self.Wlda = 0
        self.Sb = 0
        self.SB_RANK = 0
        self.Sw = 0
        self.SW_RANK = 0
        self.ref_coeffs = 0



    def train(self, training_data, PCA_unit):

        class_means = compute_class_means(training_data)
        class_scatters = compute_class_scatters(training_data, class_means)
        self.Sb = compute_Sb(class_means)
        self.SB_RANK = np.linalg.matrix_rank(self.Sb)  # Rank is c - 1 -> 51
        self.Sw = compute_Sw(class_scatters)
        self.SW_RANK = np.linalg.matrix_rank(
            self.Sw)  # Rank is N - c -> 312(train_imgs) - 52 = 260 (same as PCA reduction
        # projection)
        Sw = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sw), PCA_unit.Wpca)
        Sb = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sb), PCA_unit.Wpca)
        S = np.matmul(np.linalg.inv(Sw), Sb)
        eig_vals, fisherfaces = find_eigenvectors(S, how_many=-1)
        M_LDA = count_non_zero(
            eig_vals) + M_LDA_reduction  # hyperparameter Mlda <= c-1 -> there should be 51 non_zero eiganvalues
        # print(Mlda)     # Mlda = c - 1 = 51
        fisherfaces_reduced = fisherfaces[:, :M_LDA]
        faces = find_projection(self.Wpca, self.training_data)
        self.ref_coeffs = self.__call__(fisherfaces_reduced, faces)

    def __call__(self, reduced_faces):
        coeffs = np.matmul(reduced_faces.transpose(), self.Wlda).transpose()
        return coeffs









class unit():

    def __init__(self, training_data):

        self.training_data = training_data

        self.PCA_unit = PCA_unit()
        self.PCA_unit.train(self.training_data)
        self.LDA_unit = LDA_unit()
        self.LDA_unit.train(self.training_data, self.PCA_unit)

    def classify_data(self, test_data):

        PCA_images = self.PCA_unit(test_data)
        LDA_coeffs = self.LDA_unit(PCA_images)  # 51 vector

        distances = []
        for i in range(LDA_coeffs.shape[1]):
            distances.append(np.linalg.norm(self.ref_coeffs - LDA_coeffs[:, i][:, None], axis=0))

        return np.floor(np.argmin(np.array(distances), axis=1) / TRAINING_SPLIT).astype(np.uint16)

class Dataset():
    # Dataset with the capability of creating bags of sub datasets

    def __init__(self, data):

        self.data = data
        self.ground_truth = self.create_label()
        self.N = data.shape[1]

    def get_bag(self, n):

        chosen_sample_indexes = np.random.randint(0, self.N, (n,))
        data = self.data[:, chosen_sample_indexes]
        ground_truth = self.ground_truth[chosen_sample_indexes]
        bag = Bag(data, ground_truth)
        return bag

    @staticmethod
    def create_label():
        true_individual_index = np.arange(0, NUMBER_PEOPLE)
        true_individual_index = np.repeat(true_individual_index[:, None], TRAINING_SPLIT, axis=1).reshape(-1)
        return true_individual_index


class Bag():

    # Bag of data that can be iterated upon on a class-basis.
    # When iterating on it, returns the class number and the corresponding data
    # Can also get the complete data without class labelling by calling get_all()

    def __init__(self, data, ground_truth):
        self.current = 0
        sort_index = np.argsort(ground_truth)
        self.data, self.ground_truth = data[:, sort_index], ground_truth[sort_index]
        self.data_by_class = []
        self.represented_classes = set([i for i in self.ground_truth])

        for i in range(NUMBER_PEOPLE):
            addition = []

            for x in [index for index, e in enumerate(self.ground_truth) if e == i]:

                addition.append(self.data[:, x])

            self.data_by_class.append(addition)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        while self.current not in self.represented_classes:
            if self.current < NUMBER_PEOPLE - 1:
                self.current += 1

            else:

                raise StopIteration
        else:
            self.current += 1
            return self.current-1, np.array(self.data_by_class[self.current-1]).transpose()

    def get_all(self):

        return self.data

    def __getitem__(self, item):
        if item not in self.represented_classes:

            return None

        return np.array(self.data_by_class[item]).transpose()

    def get_dim(self):

        return self.data.shape[0]





if __name__ == '__main__':

    [training_data, testing_data], means = import_processing(INPUT_PATH)    # Training and Testing data have the
    # training mean removed
    dataset = Dataset(training_data)

    bag1 = dataset.get_bag(20)

    print(bag1.represented_classes)
    unit1 = unit(bag1)


    ''' Start classification procedure'''



