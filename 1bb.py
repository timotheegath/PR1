import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
# from memory_profiler import profile
from scipy.io import loadmat

from ex1a import find_eigenvectors, find_projection, split_data, INPUT_PATH, compute_S, recognize, NUMBER_PEOPLE, TRAINING_SPLIT, count_non_zero
from ex1b import retrieve_low_eigvecs
from ex1aa import reconstruct
from in_out import display_eigenvectors, save_values
import time
import cv2


NUMBER_OF_EIGENVECTORS = -1

def import_processing(data, class_means=False):

    faces = loadmat(data)
    # faces dimension is 2576, 520 -> each image is column vector of pixels(46, 56)
    X = np.reshape(faces['X'], (46*56, 52, 10))  # separate arrays for each person
    X = split_data(X)
    x_t = X[0]
    x_test = X[1]

    global_mean = np.mean(x_t, axis=1)
    data = [(x - global_mean[..., None]) for i, x in enumerate(X)]

    training_data = x_t
    classy_means = [np.mean(x_t[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT], axis=1) for i in range(NUMBER_PEOPLE)]
    for i in range(NUMBER_PEOPLE):
        training_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT] = x_t[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT] - classy_means[i][:, None]
    classy_data = [training_data, X[1]]

    return data, global_mean, classy_data, classy_means


def identify_failure(bool_a, number=-1):

    indices = np.argwhere(~bool_a)[:, 0]  # Gives original indices after sorting

    return indices[:number]


def identify_success(bool_a, number=-1):

    indices = np.argwhere(bool_a)[:, 0]  # Gives original indices after sorting

    return indices[:number]

# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



def bool_and_accuracy(ground_truth, prediction):

    correct = ground_truth == prediction
    accuracy = (correct[correct].shape[0]) / (ground_truth.shape[0])
    return correct, accuracy


def create_ground_truth():

    true_individual_index = np.arange(0, NUMBER_PEOPLE)
    true_individual_index = np.repeat(true_individual_index[:, None], 10-TRAINING_SPLIT, axis=1).reshape(-1)
    return true_individual_index



def classify(projections_training, projections_test):

    distances = []
    for i in range(projections_test.shape[1]):
        distances.append(np.linalg.norm(projections_training - projections_test[:, i][:, None], axis=0))
    return np.floor(np.argmin(np.array(distances), axis=1)/TRAINING_SPLIT).astype(np.uint16)



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
    NN_accuracy = []
    REC_accuracy = []
    NN_durations = []
    REC_durations = []
    # parameter_list = range(10, 363, 20)
    parameter_list = [363]
    DISPLAY = True
    for how_many_eigenvectors in parameter_list:
        NN = True
        # how_many_eigenvectors = -1
        [training_data, testing_data], glob_mean, [classy_train_data, classy_test_data], classy_means = import_processing(INPUT_PATH)
        # if NN:
            # [training_data, testing_data], means = import_processing(INPUT_PATH)
        eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), how_many_eigenvectors)
        eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
        projections_training, projections_test = find_projection(eigenvectors, training_data),\
                                                 find_projection(eigenvectors, testing_data)
        t1 = time.time()
        recognised_faces = classify(projections_training, projections_test)
        t2 = time.time()
        true_faces = create_ground_truth()

        bool_recognised, acc = bool_and_accuracy(true_faces, recognised_faces)
        NN_accuracy.append(acc)
        conf_matrix = confusion_matrix(true_faces, recognised_faces)
        if DISPLAY:
            # plot_confusion_matrix(conf_matrix, classes=np.arange(0, NUMBER_PEOPLE), normalize=True)
            failures = identify_failure(bool_recognised)

            display_eigenvectors(testing_data[:, failures] + glob_mean[:, None], eig=False)
            success = identify_success(bool_recognised)
            display_eigenvectors(testing_data[:, success] + glob_mean[:, None], eig=False)
        print('Accuracy of', acc)
        # name = 'results_' + 'NN'
        duration = t2 - t1
        NN_durations.append(duration)



        # else:
            # [training_data, testing_data], means = import_processing(INPUT_PATH, class_means=True)
        classy_eigenvectors = []
        for i in range(NUMBER_PEOPLE):
            eigv, eigvec = find_eigenvectors(compute_S(classy_train_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT],
                                                       low_res=True), -1)
            eigvec = retrieve_low_eigvecs(eigvec, classy_train_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT])
            # no_non_zero = count_non_zero(eigv)
            eigvec = eigvec[:, :how_many_eigenvectors]
            classy_eigenvectors.append(eigvec)
        t1 = time.time()
        classifications = classify_Rec(classy_test_data, classy_eigenvectors, classy_means)
        t2 = time.time()
        true_faces = create_ground_truth()
        bool_recognised, acc = bool_and_accuracy(true_faces, classifications)
        REC_accuracy.append(acc)
        conf_matrix = confusion_matrix(true_faces, classifications)
        print('Accuracy of', acc)
        # name = 'results_' + 'rec'
        if DISPLAY:
            # plot_confusion_matrix(conf_matrix, classes=np.arange(0, NUMBER_PEOPLE), normalize=True)

            failures = identify_failure(bool_recognised)
            # for i in range(NUMBER_PEOPLE):
            #     classy_test_data[:, i * (10 - TRAINING_SPLIT):(i + 1) * (10 - TRAINING_SPLIT)] = classy_test_data[:, i * (10 - TRAINING_SPLIT):(i + 1) * (10 - TRAINING_SPLIT)] + classy_means[i][:, None]
            display_eigenvectors(classy_test_data[:, failures], eig=False)
            success = identify_success(bool_recognised)
            display_eigenvectors(classy_test_data[:, success], eig=False)



        duration = t2-t1
        REC_durations.append(duration)

        name = 'results_' + 'NN_' + 'REC'
        save_values({'NN_accuracy': np.array(NN_accuracy), 'REC_accuracy': np.array(REC_accuracy), 'NN_duration': np.array(NN_durations), 'REC_duration': np.array(REC_durations), 'n_eigenvecs': np.array(parameter_list)}, name=name)
    
    # print(recognised_faces)
    # print(bool_recognised)
