function softmaxSolver_MNIST

clear all
close all
clc



if 1

%% load data
images = loadMNISTImages('train-images.idx3-ubyte'); 
labels = loadMNISTLabels('train-labels.idx1-ubyte'); 

%% train
nclass = 10;
y_train = labels + 1.0;
x_train = images';
clear images
clear lables

% softmax
option.max_itr = 500;
option.min_eps = 1e-3;
option.C = 0.01;
options.epochs = 5; 
options.minibatch = 100;
options.alpha = 1e-1;
options.momentum = .95;
% FastDescent ConjugateGradient Newton FixedNewton DFP BFGS SGD
[theta minJ] = softmaxLBFGS(x_train, y_train, option);
save theta.mat theta
clear x_train
clear y_train
end

%% test
images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
load theta.mat

y_test = labels + 1.0;
x_test = [ones(length(y_test), 1), images'];
clear images
clear lables

% hypothesis
p = softmaxFunc(x_test, theta);
[Y I] = max(p, [], 2);
acc = sum(I==y_test)/length(y_test)