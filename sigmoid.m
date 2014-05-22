function h = sigmoid(x, theta)
%Sigmoid function || Logistic function
h = 1.0 ./ (1.0 + exp(-x*theta));