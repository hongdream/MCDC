function [label, W] = GAME(K, data, seed)
%GAME
%   input:
%   K: the number of clusters
%   data: data set
%   seed: seed point
%   output:
%   label: the label of data
%   W: the weight of attribute
[N, D] = size(data);
% initial center of clusters
C = data(seed, :);
Times = 100;

% initialization 
label = zeros(N, 1);
label_old = zeros(N, 1);
W = ones(1, D);
%% Partition
convergence = 0;
while convergence == 0
for iter = 1:Times      
    % assign
    for i = 1:N
        distance = data(i, :) ~= C;
        distance = distance .* repmat(W, K, 1);
        distance = sum(distance, 2);
        [~, idx] = min(distance);
        label(i) = idx;
    end
    % update
    for k = 1:K
        Centroid(k, :) = mode(data(label == k, :));
    end
    if isequal(C, Centroid)
        break;
    end
    
    C = Centroid;
end
if isequal(label_old, label)
    convergence = 1;
    break;
end
%% update the weight of attribute
W = Weight(data, label, Centroid);
label_old = label;

end
end

