function h = softmaxFunc(x, theta)
% softmax hypothesis function 
% x      -- input data, size = [m, n], m:samples number, n:feature dimension;
% theta  -- parameters, size = [n, k], n:elements nubmer, k:class number;
% h      -- hypothesis, size = [m, k]
% author -- amadeuzou AT gmail
% date   -- 11/14/2013, Beijing, China

k = size(theta, 2);
h = exp(x*theta);
h = h./repmat(sum(h, 2), 1, k);