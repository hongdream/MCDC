function W = Weight(data, label, C)
%C_WEIGHT Calculate the weight of each attribute
%data: categorical data
%K: number of cluster
%label: label of k-modes

%% initialization
[N, D] = size(data);
similarity = zeros(1, D);
for i = 1:N
    distance = data(i, :) == C(label(i), :);
    similarity = similarity + distance;
end
similarity = similarity ./ N;
W = similarity ./ sum(similarity);
end

