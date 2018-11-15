import numpy as np
import scipy.io
import time
from scipy.io import loadmat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from ex1a import count_non_zero
from in_out import display_eigenvectors, save_values

DEFAULT_WLDA = np.zeros((2576, 1))
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




def compute_Sw(training_bag, class_means):

    class_scatters = np.zeros((training_bag.get_dim(), training_bag.get_dim()),
                              dtype=np.float32)
    for c, data in training_bag:

        meaned_training_data = (data - class_means[:, c][:, None]).transpose()[:, None]
        # scatter = np.zeros(class_scatters.shape)
        # for ci in range(meaned_training_data.shape[1]):
        class_scatters += np.sum(np.matmul(meaned_training_data, meaned_training_data.transpose(0, 2, 1)), axis=0)

    return class_scatters


def compute_Sb(class_means):

    global_mean = np.mean(class_means, axis=1, keepdims=True)
    global_mean = np.repeat(global_mean, NUMBER_PEOPLE, axis=1)
    Sb = np.matmul(class_means - global_mean, (class_means - global_mean).transpose())

    return Sb

class Ensemble():

    def __init__(self, n, training_data, bag_size):
        self.n = n
        self.units = []
        for i in range(n):
            print('Creating unit', i,'...')
            self.units.append(Unit(training_data.get_bag(bag_size)))

    def classify(self, test_data, MODE='mean'):
        p_distrib = np.zeros((self.n, NUMBER_PEOPLE, test_data.shape[1]))
        for i, u in enumerate(self.units):

            p_distrib[i] = u.classify(test_data)

        if MODE is 'mean':

            p_distrib = np.mean(p_distrib, axis=0)

        return p_distrib


class PCA_unit():

    def __init__(self):
        self.Wpca = 0
        self.ref_coeffs = [[]] * NUMBER_PEOPLE



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

        M_PCA = max(training_data.shape[1]-NUMBER_PEOPLE, training_data.shape[1]) + M_PCA_reduction   # hyperparameter Mpca <= N-c
        eig_vec_reduced = eig_vec[:, :M_PCA]

        self.Wpca = eig_vec_reduced
        # display_eigenvectors(eig_vec_reduced, eig=True)  Wda ok
        leftover_set = set(range(NUMBER_PEOPLE))
        for c, data in training_bag:
            leftover_set -= set([c])
            self.ref_coeffs[c] = self.__call__(data)

            example_dim =  self.ref_coeffs[c].shape

        for c in leftover_set:
            self.ref_coeffs[c] = np.zeros(example_dim)

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
        self.ref_coeffs = [[DEFAULT_WLDA]] * NUMBER_PEOPLE



    def train(self, training_bag, PCA_unit):

        class_means = compute_class_means(training_bag)
        self.Sw = compute_Sw(training_bag, class_means).astype(np.float64)

        self.Sb = compute_Sb(class_means).astype(np.float64)
        # self.SB_RANK = np.linalg.matrix_rank(self.Sb)  # Rank is c - 1 -> 51

        # self.SW_RANK = np.linalg.matrix_rank(
        #     self.Sw)  # Rank is N - c -> 312(train_imgs) - 52 = 260 (same as PCA reduction
        # projection)
        t1 = time.time()
        self.Sw = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sw), PCA_unit.Wpca)
        t2 = time.time()

        self.Sb = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sb), PCA_unit.Wpca)
        t3 = time.time()
        print('Sw takes {} s and Sb {} s'.format(t2-t1, t3-t2))
        S = np.matmul(np.linalg.inv(self.Sw), self.Sb)
        eig_vals, fisherfaces = find_eigenvectors(S, how_many=-1)
        t4 = time.time()
        print('eigenvectors take {} s '.format(t4-t3))
        M_LDA = count_non_zero(eig_vals) + M_LDA_reduction  # hyperparameter Mlda <= c-1 -> there should be 51 non_zero

        # print('soubles', training_bag.doubles, 'SW_RANK', self.SW_RANK, 'SB_RANK', self.SB_RANK, 'classes', len(training_bag.represented_classes), 'MLDA', M_LDA)     # Mlda = c - 1 = 51
        self.Wlda = fisherfaces[:, :M_LDA]

        for c, reduced_face in enumerate(PCA_unit.ref_coeffs):
            #ref_coeffs: M_LDA X How many examples per class were available
            self.ref_coeffs[c] = self.__call__(np.array(reduced_face))
            # print(self.ref_coeffs[c].shape)
        t5 = time.time()
        print('Calling takes {} s '.format(t5-t4))
    def __call__(self, reduced_faces):
        coeffs = np.matmul(reduced_faces.transpose(), self.Wlda).transpose()
        return coeffs


class Unit():

    def __init__(self, training_data):

        self.training_data = training_data
        all_classes = set(range(NUMBER_PEOPLE))
        not_covered = all_classes - training_data.represented_classes
        print('Classes not covered in this unit: {}'.format(not_covered))
        print('PCA unit training...')
        self.PCA_unit = PCA_unit()
        self.PCA_unit.train(self.training_data)
        print('Done')
        print('LDA unit training...')
        self.LDA_unit = LDA_unit()
        self.LDA_unit.train(self.training_data, self.PCA_unit)
        print('Done')

    def classify(self, test_data):

        PCA_images = self.PCA_unit(test_data)

        LDA_coeffs = self.LDA_unit(PCA_images)  # 51 vector
        print(LDA_coeffs)

        distances = np.zeros((NUMBER_PEOPLE, LDA_coeffs.shape[1]))
        for i in range(LDA_coeffs.shape[1]):

            for c, reduced_face in enumerate(self.LDA_unit.ref_coeffs):

                distances[c, i] = np.min(np.linalg.norm(reduced_face - LDA_coeffs[:, i][:, None], axis=0))

        distances = distances/np.max(distances, axis=0, keepdims=True)
        return distances

class Dataset():
    # Dataset with the capability of creating bags of sub datasets

    def __init__(self, data):

        self.data = data
        self.ground_truth = self.create_label()
        self.N = data.shape[1]


    def get_bag(self, n):

        chosen_sample_indexes = np.random.randint(0, self.N, (n,))
        unique, counts = np.unique(chosen_sample_indexes, return_counts=True)
        doubles = 0
        for c in counts:
            if c > 1:
                doubles += c-1
        data = self.data[:, chosen_sample_indexes]
        ground_truth = self.ground_truth[chosen_sample_indexes]
        bag = Bag(data, ground_truth, doubles)
        print('{:4f}% repeats in this bag'.format(doubles/n*100))
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

    def __init__(self, data, ground_truth, doubles):
        self.current = 0
        sort_index = np.argsort(ground_truth)
        self.data, self.ground_truth = data[:, sort_index], ground_truth[sort_index]
        self.data_by_class = []
        self.represented_classes = set([i for i in self.ground_truth])
        self.doubles = doubles


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


def create_ground_truth():
    true_individual_index = np.arange(0, NUMBER_PEOPLE)
    true_individual_index = np.repeat(true_individual_index[:, None], 10 - TRAINING_SPLIT, axis=1).reshape(-1)
    return true_individual_index



if __name__ == '__main__':

    [training_data, testing_data], means = import_processing(INPUT_PATH)    # Training and Testing data have the
    # training mean removed

    g_t = create_ground_truth()
    dataset = Dataset(training_data)
    ensemble = Ensemble(10, dataset, 300)
    classification = ensemble.classify(testing_data)
    for i in range(testing_data.shape[1]):

        plt.bar(np.arange(0, NUMBER_PEOPLE), classification[:, i])
        plt.title('Class should be {}'.format(g_t[i]))
        plt.show()
        plt.pause(1)




    ''' Start classification procedure'''



