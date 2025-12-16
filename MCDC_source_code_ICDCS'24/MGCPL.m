function [granularity, representation] = MGCPL(data, K, rate)
%MGCPL 
%   input:
%   data : data set
%   K : initial number of clusters k
%   rate : learning rate in MGCPL
%   output:
%   granularity : a series of number of clusters in different granularity
%   representation : new representation of data

%% initialization
[N, D] = size(data);
X = data;

%% Iteratively find the appropriate number of clusters at different granularities
% parameters
convengence = false;   
sigma = 0;
m = max(X(:));
granularity = [];
representation = {};

while convengence == false   
    label = zeros(N, 1);
    W = zeros(K, D) + (1 ./ D);
    % Select K initial seed points by the OI algorithm - an initialization algorithm
    seed = OI(X, N, K, D);
    U = X(seed, :);
    
    % Statistical cluster center value information
    % In each cluster, the frequency of occurrence of different values under different attributes
    sample_frequency = zeros(m, D, K);      
    % The frequency of the attribute value under each object being not empty
    class_frequency = zeros(K, D);
    for j = 1:K
        for r = 1:D
            if U(j, r) ~= 0
                sample_frequency(U(j, r), r, j) = sample_frequency(U(j, r), r, j) + 1;
                class_frequency(j, r) = class_frequency(j, r) + 1;
            end
        end
    end
    class_count = ones(K, 1);
 
    %% Partition
    for i = 1:N
        x = X(i, :);
        % Calculate the similarity between object and clusters
        s = class_assign(x, sample_frequency, class_frequency, W, D, K);
        label(i) = find(max(s) == s, 1);
        % Update cluster statistics
        for r = 1:D
            if x(r) ~= 0
                sample_frequency(x(r), r, label(i)) = sample_frequency(x(r), r, label(i)) + 1;
                class_frequency(label(i), r) = class_frequency(label(i), r) + 1;
            end
        end
        class_count(label(i)) =  class_count(label(i)) + 1;
    end
    
    %% Update cluster statistics
    sample_frequency = zeros(m, D, K);
    class_frequency = zeros(K, D);
    class_count = zeros(K, 1);
    for i = 1:N
        for r = 1:D
            if X(i, r) ~= 0
                sample_frequency(X(i, r), r, label(i)) = ...
                sample_frequency(X(i, r), r, label(i)) + 1;
                class_frequency(label(i), r) = class_frequency(label(i), r) + 1;
            end
        end
        class_count(label(i)) = class_count(label(i)) + 1;
    end
    
    %% Competitive Learning Punishment Mechanism
    % parameters
    n = ones(K, 1);
    beta = ones(K, 1);
    noChange = false;
    while noChange == false
        noChange = true;
        for i = 1:N
            x = X(i, :);
            % Calculate the similarity between object and clusters
            s = class_assign(x, sample_frequency, class_frequency, W, D, K);
            % Find winners and rival
            y = n ./ sum(n);        % winning frequency
            g = 1 ./ (1 + exp(-10 .* beta + 5));       % weight
            cost = (1 - y) .* g .* s;
            [~, index] = sort(cost,"descend");
            v = index(1);       % winner
            r = index(2);       % rival
            
            % update parameter
            n(v) = n(v) + 1;
            beta(v) = beta(v) + rate;
            beta(r) = beta(r) - rate * s(r);
            label_new = v;
            label_old = label(i);
            label(i) = label_new;
            
            % Update cluster statistics
            for r = 1:D
                if x(r) ~= 0
                    sample_frequency(x(r), r, label_new) = ...
                    sample_frequency(x(r), r, label_new) + 1;
                    sample_frequency(x(r), r, label_old) = ...
                    sample_frequency(x(r), r, label_old) - 1;
                    class_frequency(label_new, r) = class_frequency(label_new, r) + 1;
                    class_frequency(label_old, r) = class_frequency(label_old, r) - 1;
                end
            end
          
            class_count(label_new, 1) = class_count(label_new, 1) + 1;
            class_count(label_old, 1) = class_count(label_old, 1) - 1;
            
            if label_new ~= label_old
                noChange = false;
            end
        end
        
        %% Update object-cluster weight
        H = zeros(K, D);
        
        % Calculation H(the contribution of feature to clusterl)
        frequency_out_class = zeros(m, D, K);
        frequency_all_class = zeros(m, D);
        for j = 1:K
            frequency_all_class = frequency_all_class + sample_frequency(:, :, j);
        end
        
        for j = 1:K
            frequency_out_class(:, :, j) = frequency_all_class - sample_frequency(:, :, j);
        end
        
        % Calculation F(inter-cluster difference) and M(intra-cluster similarity)
        for j = 1:K
            index = find(j == label);
            count = size(index, 1);
            x = X(index, :);
            for r = 1:D
                F = 0;
                M = 0;
                for t = 1:m
                    temp = ((sample_frequency(t, r, j) / class_frequency(j, r))...
                    - (frequency_out_class(t, r, j) / (N - class_frequency(j, r)))) .^ 2;
                    if isnan(temp)
                        F = F + 0;
                    else
                        F = F + temp;
                    end
                end
                F = sqrt(F) / sqrt(2);
                for i = 1:count
                    if x(i, r) ~= 0
                        M = M + sample_frequency(x(i, r), r, j) / class_frequency(j, r);
                    end
                end
                M = M / count;
                H(j, r) = F * M;
            end
        end
        
        % update H(the contribution of feature to clusterl)
        for j = 1:K
            W(j, :) = H(j, :) ./ sum(H(j, :));
        end
    end
    
    K_new = 0;
    k_array = ones(1, K);
    for j = 1:K
        if class_count(j) > 1
            K_new = K_new + 1;
        else
            % Reassign
            k_array(j) = 0;
            index = find(label == j);
            number = size(index, 1);
            for t = 1:number
                x = X(index(t), :);
                s = class_assign(x, sample_frequency, class_frequency, W, D, K);
                label_old = label(index(t));
                label(index(t)) = find(max(s(find(k_array))) == s);
                % Update cluster statistics
                for r = 1:D
                    if x(r) ~= 0
                        sample_frequency(x(r), r, label(index(t))) = sample_frequency(x(r), r, label(index(t))) + 1;
                        sample_frequency(x(r), r, label_old) = sample_frequency(x(r), r, label_old) - 1;
                        class_frequency(label(index(t)), r) = class_frequency(label(index(t)), r) + 1;
                        class_frequency(label_old, r) = class_frequency(label_old, r) - 1;
                    end
                end
                class_count(label(index(t)), 1) =  class_count(label(index(t)), 1) + 1;
                class_count(label_old, 1) = class_count(label_old, 1) - 1;
            end
        end
    end
    
    sigma = sigma + 1;
    
    if sigma > 1 && K_new == granularity(sigma - 1)
        convengence = true;
        continue;
    end

    flag = false;
    granularity(sigma) = K_new;
    if granularity(sigma) == 1
        convengence = 1;
    end
    
    if sigma > 0
        cnt = 1;
        representation{sigma} = zeros(N, 1);
        for j = 1:K
            if class_count(j) > 1
                index = find(j == label);
                representation{sigma}(index) = cnt;
                cnt = cnt + 1;
            end
        end
    end
    K = K_new;
end
end

