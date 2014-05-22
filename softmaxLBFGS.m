function [theta cost] = softmaxLBFGS(x, y, option)
% softmax Logistic Regression　Ｓolver: Limited-memory BFGS
% http://en.wikipedia.org/wiki/Limited-memory_BFGS

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
%theta = 0.001 * randn(n+1, numClass);
theta_k = theta;
p = {};
q = {};

lambda0 = 0;
step0 = 0.1;
J = [];
itr = 0;
err = 0;

% hypothesis
%h = softmaxFunc(x, theta_k);
% gradient
[cost g] = softmaxCostFunc(x, y, theta, option.C);
% descent direction
d = -g;

while(1)

    % linear search
    param.x = x;
    param.y = y;
    param.theta = theta;
    param.d = d;
    param.C = option.C;
    lamb = smLinearSearch(@softmaxCostFuncLambda, param, lambda0, step0);
    theta_k = theta + lamb.*d;
        
    %%
        [cost gk] = softmaxCostFunc(x, y, theta_k, option.C);
        pk = theta_k - theta;
        qk = gk - g;
        p = [p pk];
        q = [q qk];
        
        gg = gk;
        alpha = zeros(itr);
        for i = 1:itr
            p_i = p{i};
            q_i = q{i};
            ruo = 1/dot(p_i(:), q_i(:));
            alpha(i) = ruo*dot(p_i(:), gg(:));
            gg = gg - alpha(i)*q_i;
        end
        
        r = gg;
        for i = itr:-1:1
            p_i = p{i};
            q_i = q{i};
            ruo = 1/dot(p_i(:), q_i(:));
            beta = ruo*dot(q_i(:), r(:));
            r = r + p_i*(alpha(i)-beta);
        end
        d = -r;
        
        theta = theta_k;
        g = gk;
        itr = itr + 1;
    %%
    err = norm(pk(:));
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
