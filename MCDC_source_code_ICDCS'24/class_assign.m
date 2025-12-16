function similarity = class_assign(x, Frequency, sample_frequency, W, d, k)
%CLASS_ASSIGN 
%   input:
%   x: data
%   Frequency: In each cluster, the frequency of occurrence of different values under different attributes
%   sample_frequency: The frequency of the attribute value under each object being not empty
%   W: the weight of attribute-weight
%   d: dimensions of attribute
%   k: the number of clusters
    %% Calculate the similarity of object-cluster
    S = zeros(k, 1);
    for j = 1:k
        for r = 1:d
            if x(1, r) ~= 0
                S(j, 1) = S(j, 1) + W(j, r) * (Frequency(x(1, r), r, j) / sample_frequency(j, r));
            end
        end
    end
    for j = 1:k
        if isnan(S(j, 1))
            S(j, 1) = 0;
        end
    end
   
    %% 计算类别
    similarity = S;
end

