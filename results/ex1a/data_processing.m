close all, clear all
load('eigVal_eigVec_meanImage_nonZeroEig.mat')
eigVal = real(eigVal);
semilogy(eigVal(1:end), 'x')
title('Real part of eigenvalues')
xlabel('Eigenvalue number')
min(eigVal)
grid on


load('reconst_error_mean_reconst_error_var.mat')
reconst_error_mean