function [theta, cost] = lrSGD(x, y, options)
% Logistic Regression　Ｓolver: Stochastic Gradient Descent
% http://en.wikipedia.org/wiki/SGD
% http://ufldl.stanford.edu/tutorial/index.php/Optimization:_Stochastic_Gradient_Descent

% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% options -- options struct
%        max_itr: max iterators
%        min_eps: min eps
%        C:       penalty factor
%        debug:   show debug message

% theta  -- parameters, size = [n+1, 1], n:elements nubmer;
% cost   -- cost

% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China

if nargin == 2
    options.C = 1;
    options.max_itr = 100;
    options.min_eps = 1e-3;
    options.debug = 1;
end
if ~isfield(options, 'C')
    options.C = 1;
end
if ~isfield(options, 'max_itr')
    options.max_itr = 100;
end
if ~isfield(options, 'min_eps')
    options.min_eps = 1e-3;
end
if ~isfield(options, 'debug')
    options.debug = 1;
end
if ~isfield(options, 'epochs')
    options.epochs = 3;
end
if ~isfield(options, 'minibatch')
    options.minibatch = 50;
end
if ~isfield(options, 'alpha')
    options.alpha = 1e-1;
end
if ~isfield(options, 'momentum')
    options.momentum = .95;
end

epochs = options.epochs; 
minibatch = options.minibatch;
alpha = options.alpha;
momentum = options.momentum;
% Setup for momentum
mom = 0.5;
momIncrease = 20;

[m, n] = size(x);
x = [ones(m, 1), x];
theta = zeros(n+1, 1);
velocity = theta;
cost = 0;

J = [];
itr = 0;
err = 0;

for i = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);

    for s = 1:minibatch:(m-minibatch+1)
        
        % get next randomly selected minibatch
        mb_data = x(rp(s:s+minibatch-1), :); 
        mb_labels = y(rp(s:s+minibatch-1));
        
        [cost, grad] = lrCostFunc(mb_data, mb_labels, theta, options.C);
        velocity = mom*velocity + alpha*grad; 
        theta = theta - velocity;
        
        % increase momentum after momIncrease iterations
        if itr == momIncrease
            mom = options.momentum;
        end;
        
        % cost record
        J = [J; cost];
        err = norm(velocity);%theta - theta_k
        itr = itr + 1;

        
        if(options.debug)
            disp(['itr = ', num2str(itr), ', cost = ', num2str(cost), ', err = ', num2str(err)]);
        end
        if itr >= options.max_itr || err <= options.min_eps || norm(grad)<=options.min_eps
             return;
        end
        
    end
    
    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;
    
end

% draw cost cure
if(options.debug)
    figure(1024)
    plot(1:length(J), J, 'b-');
    xlabel('iterators');
    ylabel('cost');
end

