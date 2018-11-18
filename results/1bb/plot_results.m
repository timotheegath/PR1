clear all, close all
NN_results = load('results_NN');
rec_results = load('results_rec');

figure()
plot(NN_results.n_eigenvecs, NN_results.accuracy * 100)
grid on
hold on
plot(rec_results.n_eigenvecs, rec_results.accuracy * 100)

xlabel('Number of eigenvectors used')
ylabel('Accuracy (%)')
legend('NN', 'Reconstruction')

figure()
plot(NN_results.n_eigenvecs, NN_results.duration)
hold on
grid on
plot(rec_results.n_eigenvecs, rec_results.duration)

xlabel('Number of eigenvectors used')
ylabel('Time taken to classify (s)')
legend('NN', 'Reconstruction')
