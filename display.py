import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2


def display_single_image(X):

    X = np.reshape(X, (56, 46))
    cv2.imshow('Single face', X)
    cv2.waitKey()


def display_eigenvectors(vecs):
    # 10 images a column
    vecs = np.abs(vecs)
    how_many = vecs.shape[0]
    rows = how_many // 10
    pics = np.zeros((56*rows, 46*10), dtype=np.uint8)
    number = 0
    for i in range(rows):
        for j in range(10):
            print(np.min(vecs[number]), np.max(vecs[number]))
            pics[i:i+56, j:j+46] = np.reshape(vecs[number, :], (56, 46))/np.max(vecs[number]) * 255
            number += 1
            if number == how_many:
                break
    cv2.imshow('Eigenvectors', pics)
    cv2.waitKey()

    if __name__ is '__main__':

        """Put things here that should launch if the this script is called directly but not if it's imported"""
