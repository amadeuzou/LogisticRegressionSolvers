function softmaxSolver_Demo_mc()

% Exercise  -- multi-class softmax solver

clear all; close all; clc


%% generate data
nsamples = 100;
ds_c1 = mvnrnd ( [1,1]*2, eye(2), nsamples );
lab_c1 = ones(nsamples, 1);
ds_c2 = mvnrnd ( [-1,-1]*2, eye(2), nsamples );
lab_c2 = 2*ones(nsamples, 1);
ds_c3 = mvnrnd ( [-1.5,1.5]*3, 1.5*eye(2), nsamples );
lab_c3 = 3*ones(nsamples, 1);
ds = [ds_c1; ds_c2; ds_c3];
lab = [lab_c1; lab_c2; lab_c3];
%scatter(ds(:, 1), ds(:, 2), 'b+', 'SizeData', 200, 'LineWidth', 2);

%%  Solver
x = ds;
y = lab;
option.C = 0.01;
option.debug = 1;
option.max_itr = 200;
option.min_eps = 1e-6;
[theta, cost] = softmaxLBFGS(x, y, option)


%% Visualize Results
figure(1)

nclass = length(unique(y));
xmin = min(x(:, 1))-1;
xmax = max(x(:, 1))+1;
margin = xmin:0.1:xmax;
colors = ['r' 'g' 'b' 'y' 'k'];
stlyes = ['r' 'g' 'b' 'y' 'k'];

hold on
for c = 1:nclass
    idc = find(y==c);
    data_c = x(idc,:);
    
    scatter(data_c(:, 1), data_c(:, 2), stlyes(c),'LineWidth', 2);

    w(1) = theta(1, c);
    w(2) = theta(2, c);
    w(3) = theta(3, c);

    plot(margin, (-w(1)-margin*w(2))/w(3), colors(c), 'LineWidth', 2);


end
axis tight
hold off

%% predict
xx = [ones(size(x, 1), 1), x];
h = softmaxFunc(xx, theta);
[v p] = max(h, [], 2);
acc = sum(p==y)/length(p);
disp(['train acc: ', num2str(acc)]);
