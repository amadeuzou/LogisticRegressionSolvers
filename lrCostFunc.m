function [J, G] = lrCostFunc(x, y, theta, alpha)
% Logistic Regression cost function 
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[0 1], m:samples number;
% theta  -- parameters, size = [n, 1], n:elements nubmer;
% alpha  -- penalty factor;
% J      -- cost
% G      -- gradient, size = [n, 1];
% author -- amadeuzou AT gmail
% date   -- 11/14/2013, Beijing, China

% hypothesis
h = sigmoid(x, theta);
m = length(y);

% cost
J = (1/m)*sum(-y.*log(h) - (1-y).*log(1-h)) + alpha*sum(theta.^2)/m;
if nargout == 1
    return;
end

% gradient
G = (1/m).*x' * (y-h);