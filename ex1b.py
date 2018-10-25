import numpy as np
from ex1a import find_eigenvectors
from in_out import load_arrays


def matrix_on_column(column, matrix, n):

    # matrix = 1/n*matrix
    return matrix.dot(column)


# High dim computation (S = 1/N * A(A^t) -> D*D)
previous_data = load_arrays('1a')
high_dim_eigVal = previous_data['eigVal']
high_dim_eigVec = previous_data['eigVec']

# Low dim computation ((A^t)A -> N*N)
data = previous_data['processedData']
N = data.shape[1]

low_dim_S = data.transpose().dot(data)  # Not really S (covariance matrix) -> it's (A^t)A
low_dim_eig = find_eigenvectors(low_dim_S)  # Eig of (A^t)A
eig_Vec = np.apply_along_axis(matrix_on_column, 0, low_dim_eig[1], data, N) # eigVec(S)[i] = A * (eigVec(A^t)A)[i] ->
# applies the data matrix on each eiganvector, to get the eiganvector of S

eig_Vec = eig_Vec[..., 0:30]  # Take first 30 eigVec
eig_Val = low_dim_eig[0][0:30]  # Take first 30 eigVal

print(eig_Vec.shape)
print(high_dim_eigVec.shape)

# Shapes are good

print(eig_Val)
print(high_dim_eigVal)
# Not so good -> Sorting may be an issue (The original eiganvectors are D*D, the low dim eiganvectors are N*N)
#