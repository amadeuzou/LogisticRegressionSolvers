function [x y] = tcdataGenerator(nsamples, per, type)
% generate two classes data
% x -- dataset, size = [nsamples 2]
% y -- labels, size = [nsamples 1]
% per -- percentage of positive samples

if nargin == 1
    type = 'normal';
    per = 0.5;
end
if nargin == 2
    type = 'normal';
end

npos = round(nsamples*per);
nneg = nsamples - npos;
switch lower(type)
    case 'normal'
        % Random arrays from the normal distribution
        data_pos = [];
        data_neg = [];
        for i = 1:npos
            data_pos = [data_pos; normrnd([1 4], 1)];
        end
        for i = 1:nneg
            data_neg = [data_neg; normrnd([4 1], 1)];
        end
        labels_pos = ones(npos, 1);
        labels_neg = -ones(nneg, 1);
        x = [data_pos; data_neg];
        y = [labels_pos; labels_neg];
        
    case 'linear'
        % Linear Seperable Data in 2-dimension
        data_pos= mvnrnd ( [1,1]*2, eye(2), npos );
        labels_pos = ones(npos, 1);
        data_neg= mvnrnd ( [-1,-1]*2, eye(2), nneg);
        labels_neg = -ones(nneg, 1);
        x = [data_pos; data_neg];
        y = [labels_pos; labels_neg];
        
    otherwise
        % normal
end