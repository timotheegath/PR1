clear all, close all
M_LDAvary = load('split_7m_pca312VARY_M_LDA.mat');
M_PCAvary = load('split_7m_lda51VARY_M_PCA2.mat');

figure()
title('Accuracy with varying values of M_{LDA} and M_{PCA}')

subplot(211)

set(gca, 'XDir','reverse')
plot(M_LDAvary.M_LDA, M_LDAvary.accuracy*100)
ylim([0 100])
title('With M_{PCA} = N - c')
xlabel('M_{LDA} values')
ylabel('Accuracy (%)')
set(gca, 'XDir','reverse')

grid on
subplot(212)

plot(M_PCAvary.M_PCA, M_PCAvary.accuracy*100)
ylim([0 100])
xlabel('M_{PCA} values')
ylabel('Accuracy (%)')
set(gca, 'XDir','reverse')
title('With M_{LDA} = c - 1')
grid on


