clear all, close all
load('results_NN_REC_vs_eigenvecs');


figure()
plot(n_eigenvecs, NN_accuracy * 100)
grid on
hold on
plot(n_eigenvecs, REC_accuracy * 100)

xlabel('Number of eigenvectors used')
ylabel('Accuracy (%)')
legend('NN', 'Reconstruction')

figure()
plot(n_eigenvecs, NN_duration)
hold on
grid on
plot(n_eigenvecs, REC_duration)

xlabel('Number of eigenvectors used')
ylabel('Time taken to classify (s)')
legend('NN', 'Reconstruction')
