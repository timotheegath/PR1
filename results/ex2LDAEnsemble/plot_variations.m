close all, clear all

load('acc_time_varying_bag_size')
plot(bag_size, accuracy)
xlabel('Bag size')
grid on
ylabel('Accuracy')


