function lrSolver_MNIST

clear all
close all
clc



%images = loadMNISTImages('train-images.idx3-ubyte');size(images)
%labels = loadMNISTLabels('train-labels.idx1-ubyte');size(labels)

%% load data
[I,labels,I_test,labels_test] = readMNIST(10000);

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



[m n] = size(x_train);
model = {};
option.C = 0.01;
option.max_itr = 1000;
option.min_eps = 1e-3;
options.epochs = 5; 
options.minibatch = 200;
options.alpha = 1e-1;
options.momentum = .95;

disp('training...');
for c = 1:nclass
    disp([num2str(c), '-th loop:']);
    idc = find(y_train==c);
    yc_train = zeros(size(y_train));
    yc_train(idc) = 1;
    % irLBFGS
    [theta, cost] = lrLBFGS(x_train, yc_train, option);
    model{c} = theta;
end

clear x_train
clear y_train

%% test
y_test = double(labels_test) + 1.0;
x_test = [];
for i = 1:length(I)
    x_test = [x_test; I_test{i}(:)'];
end
x_test = [ones(size(x_test, 1), 1) im2double(x_test) ];
clear I_test
clear lables_test

accuracy = [];
disp('testing...');
for c = 1:nclass
    disp([num2str(c), '-th loop:']);
    idc = find(y_test==c);
    yc_test = zeros(size(y_test));
    yc_test(idc) = 1;
    theta = model{c};
    % predict
    h = sigmoid(x_test, theta);
    p = ones(size(h));
    p(find(h<0.5)) = 0;
    acc = sum(p==yc_test)/length(p);
    accuracy = [accuracy acc];
    disp(['accuracy: ', num2str(acc)])
end
disp(['avg-accuracy: ', num2str(mean(accuracy))])