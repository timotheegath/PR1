close all, clear all

load('acc_time_varying_bag_sizeM_LDA_is_true')
plot(bag_size, accuracy)
xlabel('Bag size')
grid on
ylabel('Accuracy')
figure()

plot(bag_size, repeats_in_bag)
xlabel('Bag size')
grid on
ylabel('Repeat percentage')

figure()


plot(M_LDA, accuracy, '*')
xlabel('M_LDA')
grid on
ylabel('Accuracy')



