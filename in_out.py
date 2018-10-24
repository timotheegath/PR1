import numpy as np
import os, sys

import __main__ as main
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import cv2


def display_single_image(X):

    X = np.reshape(X, (46, 56)).transpose()
    cv2.imshow('Single face', X)
    cv2.waitKey()


def respan_eigenface(face):
    # Face is a vector
    minimum = np.min(face)
    face -= minimum
    maximum = np.max(face)
    face /= maximum
    face *= 255
    return face.astype(np.uint8)


def display_eigenvectors(vecs):
    # 10 images a column
    vecs = np.real(vecs)
    how_many = vecs.shape[1]
    rows = how_many // 10
    pics = np.zeros((56*rows, 46*10), dtype=np.uint8)
    number = 0
    for i in range(rows):
        for j in range(10):

            pics[i*56:i*56+56, j*46:j*46+46] = np.reshape(respan_eigenface(vecs[:, number]), (46, 56)).transpose()
            number += 1
            if number == how_many:
                break
    cv2.imshow('Eigenvectors', pics)
    cv2.waitKey()
    return pics


def save_image(image_dic): # Feed a dictionary with all images in it

    caller = os.path.basename(main.__file__)[:-3]  # identify who called the function to get the exercise number, remove
    # the .py extension
    if not os.path.exists(os.path.join('results', caller)):
        print('Created folder')
        os.mkdir('results/' + caller)  # Make folders if they don't exist
    for key in image_dic:
        name = key + '.png'
        image = image_dic[key]
        path = os.path.join('results', caller, name) # Path where image will be saved

        if image.dtype is np.float32:
            if np.any(image>1):
                image = image.astype(np.uint8)
            else:
                image = (image*255).astype(np.uint8)
        cv2.imwrite(path, image)


def save_values(values_dic):  # Feed a dictionary with all arrays in it
    caller = os.path.basename(main.__file__)[:-3]  # identify who called the function to get the exercise number, remove
    # the .py extension
    if not os.path.exists(os.path.join('results', caller)):
        print('Created folder')
        os.mkdir('results/' + caller)  # Make folders if they don't exist

    name = "_".join(values_dic.keys())

    path = os.path.join('results', caller, name)  # Path where image will be saved
    savemat(path, values_dic)


def load_arrays(question_number): # Load all results from a previous question
    path = os.path.join('results', 'ex{}'.format(question_number))
    out_dic = {}
    files = os.listdir(path)  # List all the files in the directory
    mat_files = [f for f in files if '.mat' in f] # only keep mat files
    for f in mat_files:
        loadmat(os.path.join(path, f), mdict=out_dic)
    return out_dic


if __name__ == '__main__':

    """Put things here that should launch if the this script is called directly but not if it's imported"""

