import numpy as np
import scipy.io
import time
from scipy.io import loadmat, savemat
from scipy import spatial
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means
import matplotlib.pyplot as plt

from in_out import display_eigenvectors, save_values

DEFAULT_WLDA = np.zeros((2576, 1))
INPUT_PATH = 'data/face.mat'
parameters = {'split': 7, 'n_units': 8, 'M_PCA': False, 'M_LDA': False, 'bag_size': 200, 'combination': 'product', 'PCA_reduction': 0, 'LDA_reduction': 0}
# A true value for MLDA and MPCA randomizes their values to be between 1/4 and 4/4 of their original value
# The combination defines how the units' outputs are combined. For now, only mean is implemented but product needs to
# be implemented

TRAINING_SPLIT = parameters['split']
NUMBER_PEOPLE = 52

T_TRAINING = 0



def count_non_zero(eigenvalues):
    temp_eig_vals = np.abs(np.real(eigenvalues))
    temp_eig_vals /= np.max(temp_eig_vals)
    # boolean_mask = eigenvalues.nonzero()  # Mask of same shape as vector which is True if value is non zero
    boolean_mask = temp_eig_vals > 0.00001 # not really non-zero
    remaining_values = eigenvalues[boolean_mask]  # Only keep non-zero values
    return remaining_values.shape[0]  # How many left ?


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
    indices = np.flip(np.argsort(np.abs(eigvalues)), axis=0) # Gives original indices after sorting
    sorted_eigvalues = eigvalues[indices]
    sorted_eigvectors = eigvectors[:, indices]

    return sorted_eigvalues[0:how_many], sorted_eigvectors[:, 0:how_many]


def find_projection(eigenvectors, faces):  # eigenvectors and faces in vector form

    coeffs = np.matmul(faces.transpose(), eigenvectors).transpose()
    # number_of_eigenvectors X Faces
    return coeffs


class Ensemble():

    def __init__(self, training_data, **kwargs):
        global T_TRAINING
        T_TRAINING = 0
        self.n = n = parameters['n_units']
        self.units = []
        bag_size = parameters['bag_size']
        if kwargs.get('load', False):
            path = 'results/ex2LDAEnsemble/'
            Wpcas = np.load(path + 'Wpca')
            Wldas = np.load(path + 'Wlda')
            for i in range(n):
                print('Creating unit', i,'...')
                self.units.append(Unit(training_data.get_bag(bag_size, Wpca=Wpcas[i], Wlda=Wldas[i])))

        else :
            for i in range(n):
                print('Creating unit', i,'...')

                bag = training_data.get_bag(bag_size)
                t_wasted = time.time()
                # FOr low bag size, it takes time to find a combination which includes all classes.
                # Remove this time from training
                self.units.append(Unit(bag))
                T_TRAINING += (time.time() - t_wasted)

    def classify(self, test_data):
        p_distrib = np.zeros((self.n, NUMBER_PEOPLE, test_data.shape[1]))
        for i, u in enumerate(self.units):

            p_distrib[i] = u.classify(test_data)

        if parameters['combination'] is 'mean':

            p_distrib = np.mean(p_distrib, axis=0)

        elif parameters['combination'] is 'product':

            p_distrib = np.prod(p_distrib, axis=0)

        p_distrib /= np.sum(p_distrib, axis=0, keepdims=True)

        return 1 - p_distrib

    def save(self):
        Wpcas = np.zeros((self.n,) + self.units[0].PCA_unit.Wpca.shape, np.complex64)
        Wldas = np.zeros((self.n,) + self.units[0].LDA_unit.Wlda.shape, np.complex64)
        for i, u in enumerate(self.units):

            Wpcas[i] = u.PCA_unit.Wpca
            Wldas[i] = u.LDA_unit.Wlda
        savemat('results/ex2LDAEnsemble/weights_'+ '_'.join('{}'.format(p) for _, p in parameters.items()),
                {'Wpca': Wpcas, 'Wlda': Wldas})

    def get_repeats(self):
        repeats = np.empty((len(self.units)))
        for i, u in enumerate(self.units):
            repeats[i] = u.repeats

        return repeats

    def get_M_LDA(self):
        M_LDA = np.empty((len(self.units)))
        for i, u in enumerate(self.units):
            M_LDA[i] = u.LDA_unit.M_LDA

        return M_LDA



class PCA_unit():

    def __init__(self):

        self.Wpca = 0
        self.ref_coeffs = [[]] * NUMBER_PEOPLE



    def train(self, training_bag):
        def retrieve_low_eigvecs(low_eigvecs, data):  # Returns normalized eigenvectors

            vecs = np.matmul(data, low_eigvecs)
            vecs /= np.linalg.norm(vecs, axis=0)[None, :]
            return vecs

        def reduce_by_PCA(training_data, means):

            training_data_norm = training_data - means
            low_S = compute_S(training_data_norm, low_res=True)
            eig_val, eig_vec = find_eigenvectors(low_S, how_many=-1)
            eig_vec = retrieve_low_eigvecs(eig_vec, training_data_norm)
            M_PCA = training_data_norm.shape[1] - NUMBER_PEOPLE + parameters['PCA_reduction']
            M_PCA -= parameters['M_PCA']*np.random.randint(int(-3*M_PCA/4), 0)# hyperparameter Mpca <= N-c
            print('M_PCA: ', M_PCA)
            eig_vec_reduced = eig_vec[:, :M_PCA]
            return eig_vec_reduced

        training_data = training_bag.get_all()
        eig_vec_reduced = reduce_by_PCA(training_data, training_bag.global_mean)
        self.Wpca = eig_vec_reduced
        self.find_ref_coeffs(training_bag)

    def find_ref_coeffs(self, training_bag):

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
        self.ref_coeffs = [[DEFAULT_WLDA]] * NUMBER_PEOPLE
        self.M_LDA = 0



    def train(self, training_bag, PCA_unit):

        def compute_Sw(training_bag):

            class_scatters = np.zeros((training_bag.get_dim(), training_bag.get_dim()),
                                      dtype=np.float64)
            for c, data in training_bag:
                meaned_training_data = (data - training_bag.class_means[:, c][:, None]).transpose()[..., None]

                class_scatters += np.sum(np.matmul(meaned_training_data, meaned_training_data.transpose(0, 2, 1)),
                                         axis=0)

            return class_scatters

        def compute_Sb(class_means):

            global_mean = np.mean(class_means, axis=1, keepdims=True)
            global_mean = np.repeat(global_mean, NUMBER_PEOPLE, axis=1)
            Sb = np.matmul(class_means - global_mean, (class_means - global_mean).transpose())

            return Sb

        class_means = training_bag.class_means
        self.Sw = compute_Sw(training_bag).astype(np.float64)

        self.Sb = compute_Sb(class_means).astype(np.float64)
        # self.SB_RANK = np.linalg.matrix_rank(self.Sb)  # Rank is c - 1 -> 51
        #
        # self.SW_RANK = np.linalg.matrix_rank(
        #     self.Sw)  # Rank is N - c -> 312(train_imgs) - 52 = 260 (same as PCA reduction
        # # projection)

        self.Sw = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sw), PCA_unit.Wpca)

        self.Sb = np.matmul(np.matmul(PCA_unit.Wpca.transpose(), self.Sb), PCA_unit.Wpca)
        S = np.matmul(np.linalg.inv(self.Sw), self.Sb)
        eig_vals, fisherfaces = find_eigenvectors(S, how_many=-1)
        eig_vals = np.real(eig_vals)
        self.M_LDA = NUMBER_PEOPLE-1 + parameters['LDA_reduction']  # hyperparameter Mlda <= c-1 -> there should be 51 non_zero
        self.M_LDA -= parameters['M_LDA'] * np.random.randint(int(-3*self.M_LDA/4), 0)
        print('M_LDA :', self.M_LDA)

        self.Wlda = fisherfaces[:, :self.M_LDA]
        self.find_ref_coeffs(PCA_unit)

    def find_ref_coeffs(self, PCA_unit):

        for c, reduced_face in enumerate(PCA_unit.ref_coeffs):
            #ref_coeffs: M_LDA X How many examples per class were available
            self.ref_coeffs[c] = self.__call__(np.array(reduced_face))

    def __call__(self, reduced_faces):
        coeffs = np.matmul(reduced_faces.transpose(), self.Wlda).transpose()
        return coeffs


class Unit():

    def __init__(self, training_data, **kwargs):

        self.training_data = training_data
        self.PCA_unit = PCA_unit()
        self.LDA_unit = LDA_unit()
        if 'Wpca' in kwargs.keys() & 'Wlda' in kwargs.keys():
            print('Loading previous weights...')
            self.PCA_unit.Wpca = kwargs['Wpca']
            self.LDA_unit.Wlda = kwargs['Wlda']
            self.PCA_unit.find_ref_coeffs(training_data)
            self.LDA_unit.find_ref_coeffs(self.PCA_unit)
            print('Done')
        else:
            print('PCA unit training...')

            self.PCA_unit.train(self.training_data)
            print('Done')
            print('LDA unit training...')

            self.LDA_unit.train(self.training_data, self.PCA_unit)
            print('Done')
        self.repeats = training_data.doubles
        self.training_data = []
    def classify(self, test_data):

        PCA_images = self.PCA_unit(test_data)

        LDA_coeffs = self.LDA_unit(PCA_images)  # 51 vector


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
        do_again = True

        while(do_again):

            chosen_sample_indexes = np.random.randint(0, self.N, (n,))

            unique, counts = np.unique(chosen_sample_indexes, return_counts=True)
            doubles = 0
            for c in counts:
                if c > 1:
                    doubles += c-1
            data = self.data[:, chosen_sample_indexes]
            ground_truth = self.ground_truth[chosen_sample_indexes]
            represented_classes = set([i for i in ground_truth])
            if not set(range(NUMBER_PEOPLE)) - represented_classes:
                do_again = False
        bag = Bag(data, ground_truth, doubles/n)

        print('{:2.1f}% repeats in this bag'.format(doubles/n*100))
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

        self.class_means = np.zeros((data.shape[0], NUMBER_PEOPLE))
        for i in range(NUMBER_PEOPLE):
            addition = []
            class_exists = False

            for x in [index for index, e in enumerate(self.ground_truth) if e == i]:
                class_exists = True
                addition.append(self.data[:, x])
            if class_exists:
                addition = np.array(addition).transpose()

                self.class_means[:, i] = np.mean(addition, axis=1)

            self.data_by_class.append(addition)
        self.global_mean = np.mean(self.class_means, axis=1, keepdims=True)

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
            return self.current-1, np.array(self.data_by_class[self.current-1])

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

    varying_parameter = 'bag_size'
    parameter_values = np.arange(100, 500, 75)
    training_times = np.zeros_like(parameter_values).astype(np.float32)
    testing_times = np.zeros_like(parameter_values).astype(np.float32)
    accuracies = np.zeros_like(parameter_values).astype(np.float32)
    repeats = np.zeros((parameter_values.shape[0], parameters['n_units']))
    M_LDAs = np.zeros((parameter_values.shape[0], parameters['n_units']))

    for nn in range(parameter_values.shape[0]):
        [training_data, testing_data], means = import_processing(INPUT_PATH)  # Training and Testing data have the
        # training mean removed
        parameters[varying_parameter] = parameter_values[nn]

        g_t = create_ground_truth()

        dataset = Dataset(training_data)
        
        ensemble = Ensemble(dataset)
        t_train = T_TRAINING

        t0 = time.time()
        classification = ensemble.classify(testing_data)
        final_class = np.argmax(classification, axis=0)
        t_class = time.time()

        training_times[nn], testing_times[nn] = t_train, t_class-t0

        def bool_and_accuracy(ground_truth, prediction):
            correct = ground_truth == prediction
            accuracy = (correct[correct].shape[0]) / (ground_truth.shape[0])

            return correct, accuracy

        _, acc = bool_and_accuracy(g_t, final_class)
        accuracies[nn] = acc
        repeats[nn] = ensemble.get_repeats()
        M_LDAs[nn] = ensemble.get_M_LDA()
        print('Accuracy :', accuracies[nn])
        # ensemble.save()
        merged_dict = {varying_parameter: parameter_values, 'accuracy': accuracies, 'training_times': training_times,
                       'testing_times': testing_times, 'repeats_in_bag':  repeats, 'M_LDA': M_LDAs}
        save_values(merged_dict, 'acc_time_varying_' + varying_parameter + parameters['combination'] + 'M_LDA_is_true')




