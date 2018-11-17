import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from memory_profiler import profile

from in_out import display_eigenvectors, display_single_image, save_image, save_values, load_arrays
from ex1a import find_eigenvectors, find_projection, import_processing, INPUT_PATH, compute_S, recognize, NUMBER_PEOPLE, TRAINING_SPLIT, count_non_zero
from ex1b import retrieve_low_eigvecs
from ex1aa import reconstruct
from in_out import display_eigenvectors
import time
import cv2


NUMBER_OF_EIGENVECTORS = -1


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


@profile
def classify(projections_training, projections_test):

    distances = []
    for i in range(projections_test.shape[1]):
        distances.append(np.linalg.norm(projections_training - projections_test[:, i][:, None], axis=0))
    return np.floor(np.argmin(np.array(distances), axis=1)/TRAINING_SPLIT).astype(np.uint16)


@profile
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
    NN = True


    if NN:
        [training_data, testing_data], means = import_processing(INPUT_PATH)
        eigenvalues, eigenvectors = find_eigenvectors(compute_S(training_data, low_res=True), -1)
        eigenvectors = retrieve_low_eigvecs(eigenvectors, training_data)
        projections_training, projections_test = find_projection(eigenvectors, training_data),\
                                                 find_projection(eigenvectors, testing_data)
        t1 = time.time()
        recognised_faces = classify(projections_training, projections_test)
        t2 = time.time()
        true_faces = create_ground_truth()
    
        bool_recognised, accuracy = bool_and_accuracy(true_faces, recognised_faces)

        conf_matrix = confusion_matrix(true_faces, recognised_faces)
        plot_confusion_matrix(conf_matrix, classes=np.arange(0, NUMBER_PEOPLE), normalize=True)
        failures = identify_failure(bool_recognised)

        display_eigenvectors(testing_data[:, failures]+means[0][:, None])
        success = identify_success(bool_recognised)
        display_eigenvectors(testing_data[:, success] + means[0][:, None])
        print(accuracy)
        name = 'results_' + 'NN'
    
        #cv2.imshow('Confusion matrix', conf_matrix)
        # cv2.waitKey()
        # print(np.unique(conf_matrix))
    else:
        [training_data, testing_data], means = import_processing(INPUT_PATH, class_means=True)
        eigenvectors = []
        for i in range(NUMBER_PEOPLE):
            eigv, eigvec = find_eigenvectors(compute_S(training_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT],
                                                       low_res=True), -1)
            eigvec = retrieve_low_eigvecs(eigvec, training_data[:, i*TRAINING_SPLIT:(i+1)*TRAINING_SPLIT])
            no_non_zero = count_non_zero(eigv)
            eigvec = eigvec[:, :no_non_zero]
            eigenvectors.append(eigvec)
        t1 = time.time()
        classifications = classify_Rec(testing_data, eigenvectors, means)
        t2 = time.time()
        true_faces = create_ground_truth()
        bool_recognised, accuracy = bool_and_accuracy(true_faces, classifications)
        conf_matrix = confusion_matrix(true_faces, classifications)
        plot_confusion_matrix(conf_matrix, classes=np.arange(0, NUMBER_PEOPLE), normalize=True)
        print(accuracy)
        name = 'results_' + 'rec'

    duration = t2-t1

    print(duration)

    save_values({'accuracy': np.array(accuracy), 'duration': duration}, name=name)
    
    # print(recognised_faces)
    # print(bool_recognised)
