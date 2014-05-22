function [J G] = softmaxCostFunc(x, y, theta, alpha)
% softmax hypothesis function 
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% y      -- labels data, size = [m, 1], values=[1 k], m:samples number, k:class number;
% theta  -- parameters, size = [n, k], n:elements nubmer, k:class number;
% alpha  -- penalty factor;
% G      -- gradient, size = [n, k];
% author -- amadeuzou AT gmail
% date   -- 11/14/2013, Beijing, China

k = size(theta, 2);
[m, n] = size(x);
H = exp(x*theta);

%% cost
M = repmat(sum(H, 2), 1, k);
Y = repmat(y, 1, k);
I = repmat(1:k, m, 1);
J = (Y==I).*log(H./M);
J = (-1/m)*sum(J(:)) + 0.5*alpha*sum(theta(:).^2);

if nargout == 1
    return;
end

%% gradient
M = repmat(sum(H, 2), 1, n);
G = [];
for c = 1:k
    p = exp(x*repmat(theta(:,c), 1, n))./M;
    Y = (y==c);
    Y = repmat(Y, 1, n) - p;
    g = x.*Y;
    g = (-1/m)*sum(g, 1) + alpha*theta(:,c)';
    G = [G g'];
end

