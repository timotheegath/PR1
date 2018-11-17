close all
clear all
load('distortion_example_image_checkpoints.mat')

figure()
semilogy(distortion, 'x')
grid on 
xlabel('Number of eigenvectors used for reconstruction')
ylabel('Distortion (SI)')