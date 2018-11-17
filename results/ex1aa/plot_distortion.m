close all
clear all
load('distortion.mat')

figure()
plot(distortion, 'x')
grid on 
xlabel('Number of eigenvectors used for reconstruction')
ylabel('Distortion (SI)')