function lrSolver_Demo_tc()

% Exercise  -- Logistic Regression Solver

clear all; close all; clc

%% generate data
nsamples = 200;
% training data
[x, y] = tcdataGenerator(nsamples, 0.5, 'normal');
y(find(y==-1)) = 0;
% testing data
[xt, yt] = tcdataGenerator(nsamples, 0.5, 'normal');
yt(find(yt==-1)) = 0;


%%  Logistic Regression Solver
% FastDescent ConjugateGradient Newton FixedNewton DFP BFGS SGD
option.C = 1;
option.debug = 1;
options.epochs = 3; 
options.minibatch = 200;
options.alpha = 1e-1;
options.momentum = .95;
[theta, cost] = lrSGD(x, y, option)

%% Visualize Results
figure(1)
subplot(121)
xmin = min(x(:, 1))-1;
xmax = max(x(:, 1))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==0),:);

scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
hold on
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight


margin = xmin:0.1:xmax;
plot(margin, (-theta(1)-margin*theta(2))/theta(3), 'r-', 'LineWidth', 2);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
option.C = 1;
[theta_, cost] = lrLBFGS(x, y, option)

%% Visualize Results
figure(1)
subplot(122)
xmin = min(x(:))-1;
xmax = max(x(:))+1;
data_pos = x(find(y==1),:);
data_neg = x(find(y==0),:);

scatter(data_pos(:, 1), data_pos(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);
hold on
scatter(data_neg(:, 1), data_neg(:, 2), 'gx', 'SizeData', 200, 'LineWidth', 2);
axis tight


margin = xmin:0.1:xmax;
plot(margin, (-theta(1)-margin*theta(2))/theta(3), 'r-', 'LineWidth', 2);
hold off

%% predict
xx = [ones(size(x, 1), 1), x];
h = sigmoid(xx, theta);
p = ones(size(h));
p(find(h<0.5)) = 0;
acc = sum(p==y)/length(p);
disp(['train acc: ', num2str(acc)]);

h = sigmoid(xx, theta);
p = ones(size(h));
p(find(h<0.5)) = 0;
acc = sum(p==yt)/length(p);
disp(['test acc: ', num2str(acc)]);

