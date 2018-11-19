clear all, close all
M_LDAvary = load('split_7m_pca312VARY_M_LDA.mat');
M_PCAvary = load('split_7m_lda51VARY_M_PCA2.mat');

figure()
title('When varying M_LDA')
plot(M_LDAvary.M_LDA, M_LDAvary.accuracy)
xlabel('M_(LDA) values')
ylabel('Accuracy')
set(gca, 'XDir','reverse')
legend('With M_(PCA) = N - c')
grid on

figure()
title('WHen varying M_PCA')
plot(M_PCAvary.M_PCA, M_PCAvary.accuracy)
xlabel('M_(PCA) values')
ylabel('Accuracy')
set(gca, 'XDir','reverse')
legend('With M_(LDA) = c - 1')
grid on


