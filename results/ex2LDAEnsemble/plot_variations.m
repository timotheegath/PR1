close all, clear all

load('acc_time_varying_PCA_reductionproduct')
plot(M_PCA(:,1), accuracy)
xlabel('M_PCA')
grid on
ylabel('Accuracy')
figure()


figure()


plot(M_LDA, accuracy, '*')
xlabel('M_LDA')
grid on
ylabel('Accuracy')
figure()
mean_corr  = squeeze(mean(corrs, 2));

stages = size(mean_corr, 1)
for i = 1:size(mean_corr, 1)
   subplot(round(stages/4), 4, i) 
   imagesc(squeeze(mean_corr(i, :, :)))
   colorbar
   daspect([1 1 1])
   title(['M_PCA: ', num2str(M_PCA(i, 1))])

end
disp(size(mean_corr))




