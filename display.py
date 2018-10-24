import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
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

    if __name__ is '__main__':

        """Put things here that should launch if the this script is called directly but not if it's imported"""
