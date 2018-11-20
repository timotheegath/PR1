close all, clear all

load('acc_time_varying_bag_size_prod (MPCA, MLDA = FALSE)')

figure()
plot(bag_size, accuracy*100)
xlabel('Bag size')
ylabel('Accuracy (%)')
title('Variation of classification accuracy against bag size')
grid on
%% 
clear all
close all
LDA_var = load('acc_time_varying_LDA_reductionproduct','-mat', 'accuracy', 'M_LDA');
PCA_var = load('acc_time_varying_PCA_reductionproduct', '-mat', 'accuracy', 'M_PCA');
plot(PCA_var.M_PCA(:,1), PCA_var.accuracy*100)
xlabel('M_{PCA}')
grid on
ylabel('Accuracy (%)')
ylim([0 100])

figure()
plot(LDA_var.M_LDA, LDA_var.accuracy*100)
xlabel('M_{LDA}')
grid on
ylabel('Accuracy (%)')
ylim([0 100])

%% 
clear all
close all
no_bag_ex = load('acc_time_no_bag_Vary_unit_paramsproduct');
bag_ex = load('acc_time_with_bag_Vary_unit_paramsproduct');
nothing = load('acc_time_no_bag_no_randVary_unit_paramsproduct');
corr_m_no_bag = squeeze(mean(no_bag_ex.corrs(1, 1, :, :), 2));
corr_m_bag = squeeze(mean(bag_ex.corrs(1, 1, :, :), 2));
corr_nothing = squeeze(mean(nothing.corrs(1, 1, :, :), 2));

label = 't';
figure()

subplot(132)
imagesc(corr_m_no_bag)
title({'Correlation between units,','no bagging, randomization of parameters'})
xlabel(label)
ylabel(label)
daspect([1 1 1])
colorbar

subplot(133)

imagesc(corr_m_bag)
xlabel(label)
ylabel(label)
colorbar
daspect([1 1 1])
title({'Correlation between units,','bagging, randomization of parameters'})

subplot(131)

imagesc(corr_nothing, [0 1])
xlabel(label)
ylabel(label)
daspect([1 1 1])
colorbar

title({'Correlation between units,','no bagging, no randomization of parameters'})

%%
clear all, close all
load('acc_time_Vary_unit_num mean')
figure()
plot(n_units, training_times)



%%
clear all, close all
load('acc_time_unit_error_analysis_no_rand')
figure()
plot(accuracy*100, 'x', 'MarkerSize',15)
hold on
grid on
plot(unit_accuracy*100, 'o', 'MarkerSize',15)
ylim([0 100])
xlim([-1 4])
xlabel('Run')
ylabel('Accuracy (%)')
legend('Mean', 'Product', 'Majority voting', 'Unit 1', 'Unit 2', 'Unit 3','Unit 4', 'Unit 5', 'Unit 6', 'Unit 7', 'Unit 8')

xticks([1 2 3])
