import numpy as np
from ex1a import find_eigenvectors, compute_S
from in_out import load_arrays
from sklearn.preprocessing import normalize


def matrix_on_column(column, matrix):

    return matrix.dot(column)


# High dim computation (S = 1/N * A(A^t) -> D*D)
previous_data = load_arrays('1a')
high_dim_eigVal = previous_data['eigVal']
high_dim_eigVec = np.real(previous_data['eigVec'])

# Low dim computation ((A^t)A -> N*N)
data = previous_data['processedData']

low_dim_S = compute_S(data.transpose())  # Not really S (covariance matrix) -> it's (A^t)A
low_dim_eig = find_eigenvectors(low_dim_S)  # Eig of (A^t)A
eig_Vec = np.apply_along_axis(matrix_on_column, 0, low_dim_eig[1], data)  # eigVec(S)[i] = A * (eigVec(A^t)A)[i] ->
# applies the data matrix on each eiganvector, to get the eiganvector of S
#eig_Vec = normalize(eig_Vec)

print(eig_Vec.shape)
print(high_dim_eigVec.shape)


print(eig_Vec[:, 0])
print(high_dim_eigVec[:, 0])

# Shapes are good

# Not so good -> Sorting may be an issue (The original eiganvectors are D*D, the low dim eiganvectors are N*N)