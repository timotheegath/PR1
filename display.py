import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

faces = loadmat('data/face.mat')
print(faces.keys())
print(faces['X'])
print(faces['X'].shape)
print(faces['l'])
print(faces['l'].shape)
print(np.unique(faces['l'], return_counts=True))

X = np.reshape(faces['X'], (46, 56, 520))

for i in range(520):

    cv2.imshow('Face {}'.format(i), cv2.resize(X[..., i].transpose(), dsize=(460, 560), interpolation=cv2.INTER_LINEAR))
    cv2.waitKey()
    cv2.destroyAllWindows()

print(X.shape)
