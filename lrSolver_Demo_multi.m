function lrSolver_Demo_multi()

% Exercise  -- Logistic Regression Solver

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

%% Logistic Regression Solver
x = ds;
y = lab;
option.C = 1;
option.debug = 0;
theta = lrSolverMulti(x, y, option);

%% Visualize Results
figure(1)

nclass = length(unique(y));
xmin = min(x(:, 1))-1;
xmax = max(x(:, 1))+1;
margin = xmin:0.1:xmax;
accuracy = [];
xx = [ones(size(x, 1), 1) x ];
colors = ['r' 'g' 'b' 'y' 'k'];
stlyes = ['r' 'g' 'b' 'y' 'k'];

hold on
for c = 1:nclass
    idc = find(y==c);
    data_c = x(idc,:);
    
    scatter(data_c(:, 1), data_c(:, 2), stlyes(c),'LineWidth', 2);

    w(1) = theta{c}(1);
    w(2) = theta{c}(2);
    w(3) = theta{c}(3);

    plot(margin, (-w(1)-margin*w(2))/w(3), colors(c), 'LineWidth', 2);

     % predict
    h = sigmoid(xx, theta{c});
    p = ones(size(h));
    p(find(h<0.5)) = 0;
    yc = zeros(size(y));
    yc(idc) = 1;
    acc = sum(p==yc)/length(p);
    accuracy = [accuracy acc];
    disp(['accuracy: ', num2str(acc)])
end
axis tight
hold off

%%
function theta = lrSolverMulti(x, y, option)

nclass = length(unique(y));
theta = [];
for c = 1:nclass
    idc = find(y==c);
    yc_train = zeros(size(y));
    yc_train(idc) = 1;
    % irLBFGS
    theta{c} = lrLBFGS(x, yc_train, option);
end