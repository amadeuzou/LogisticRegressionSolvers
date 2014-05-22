function softmaxSolver_MNIST_32

clear all
close all
clc


%% load data
[I,labels,I_test,labels_test] = readMNIST(10000);


if 1


%% train
nclass = 10;
y_train = double(labels) + 1.0;
x_train = [];
for i = 1:length(I)
    x_train = [x_train; I{i}(:)'];
end
x_train = im2double(x_train);
%clear I
%clear lables

% softmax
option.max_itr = 500;
option.min_eps = 1e-3;
option.C = 0.01;
options.epochs = 5; 
options.minibatch = 250;
options.alpha = 1e-1;
options.momentum = .95;
% FastDescent ConjugateGradient Newton FixedNewton DFP BFGS SGD
[theta minJ] = softmaxSGD(x_train, y_train, option);
save theta.mat theta
clear x_train
clear y_train
end

%% test
y_test = double(labels_test) + 1.0;
x_test = [];
for i = 1:length(I)
    x_test = [x_test; I_test{i}(:)'];
end
x_test = im2double(x_test);
x_test = [ones(size(x_test, 1), 1), x_test];
clear I_test
clear lables_test

load theta.mat
% hypothesis
p = softmaxFunc(x_test, theta);
[Y I] = max(p, [], 2);
acc = sum(I==y_test)/length(y_test)