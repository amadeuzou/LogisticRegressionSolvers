function [theta cost] = lrFastDescent(x, y, option)
% Logistic Regression　Ｓolver: Fast Descent
% http://en.wikipedia.org/wiki/Steepest_descent
% http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m

% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[-1 1], m:samples number;
% theta  -- parameters, size = [n+1, 1], n:elements nubmer;
% cost   -- cost
% option -- option struct
%        max_itr: max iterators
%        min_eps: min eps
%        C:       penalty factor
%        debug:   show debug message
% author -- amadeuzou AT gmail
% date   -- 11/19/2013, Beijing, China


if nargin == 2
    option.C = 1;
    option.max_itr = 100;
    option.min_eps = 1e-3;
    option.debug = 1;
end
if ~isfield(option, 'C')
    option.C = 1;
end
if ~isfield(option, 'max_itr')
    option.max_itr = 100;
end
if ~isfield(option, 'min_eps')
    option.min_eps = 1e-3;
end
if ~isfield(option, 'debug')
    option.debug = 1;
end

[m, n] = size(x);
x = [ones(m, 1), x];
theta = zeros(n+1, 1);
J = [];

lambda0 = 0;
step0 = 0.1;
itr = 0;
err = 0;
while(1)
    % gradient
    %g = (1/m).*x' * (y-h);
    [cost g] = lrCostFunc(x, y, theta, option.C);
    % descent direction
    d = -g;
    
    % linear search
    param.x = x;
    param.y = y;
    param.theta = theta;
    param.d = d;
    param.C = option.C;
    lamb = lrLinearSearch(@lrCostFuncLambda, param, lambda0, step0);
    theta = theta + lamb.*d;
   
    % cost record
    J = [J; cost];
    
    itr = itr + 1;
    err = norm(lamb*d);
    if(option.debug)
        disp(['itr = ', num2str(itr), ', cost = ', num2str(cost), ', err = ', num2str(err)]);
    end
    if itr >= option.max_itr || err <= option.min_eps || norm(g)<=option.min_eps
        break;
    end
    
end
% draw cost cure
if(option.debug)
    figure(1024)
    plot(1:length(J), J, 'b-');
    xlabel('iterators');
    ylabel('cost');
end


function cost = lrCostFuncLambda(param, lambda)

theta = param.theta + lambda.*param.d;
cost = lrCostFunc(param.x, param.y, theta, param.C);



