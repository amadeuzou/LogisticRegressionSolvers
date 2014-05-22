function [theta cost] = softmaxBFGS(x, y, option)
% softmax Logistic Regression　Ｓolver:   BFGS
% http://en.wikipedia.org/wiki/BFGS_method

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



numClass = length(unique(y));
[m, n] = size(x);
x = [ones(m, 1), x];
theta = zeros(n+1, numClass);
theta_n = theta;
J = [];
H = eye(n+1);
I = eye(n+1);

lambda0 = 0;
step0 = 0.1;
itr = 0;
err = 0;
j = 0;
% hypothesis
%h = softmaxFunc(x, theta_n);
% gradient
[cost g] = softmaxCostFunc(x, y, theta, option.C);
% descent direction
d = -H*g;
while(1)

    % linear search
    param.x = x;
    param.y = y;
    param.theta = theta;
    param.d = d;
    param.C = option.C;
    lamb = smLinearSearch(@softmaxCostFuncLambda, param, lambda0, step0);
    theta_n = theta + lamb.*d;
    
    if j<n
        j = j+1;
        continue;
    else
        [cost gk] = softmaxCostFunc(x, y, theta_n, option.C);
        p = theta_n - theta;
        q = gk - g;
        ruo = 1/dot(p(:), q(:));
        v = I - ruo*q*p';
        H = v'*H*v + ruo*p*p';
        
        theta = theta_n;
        g = gk;
        d = -H*g;
        j = 0;
        err = norm(p(:));
        itr = itr + 1;
    end
   
    J = [J; cost];
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

function J = softmaxCostFuncLambda(param, lambda)

theta = param.theta + lambda.*param.d;
k = size(theta, 2);
m = size(param.x, 1);
H = exp(param.x*theta);
M = repmat(sum(H, 2), 1, k);
Y = repmat(param.y, 1, k);
I = repmat(1:k, m, 1);
J = (Y==I).*log(H./M);
J = (-1/m)*sum(J(:)) + 0.5*param.C*sum(theta(:).^2);
