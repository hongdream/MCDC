function Ami = AMI(True_label, class_label)
%AMI adjusted mutual information
%   Calculation formula：
%   AMI = (MI - E[MI]) / ((H(C1) + H(C2))/2 - E[MI])
%   MI：Mutual Information between two clustering results
%   E[MI]：The expected Mutual Information between two clustering results when randomly assigning clusters
%   H(C1)、H(C2)：Entropy of clustering results
%   input:
%   True_label： real label of data
%   class_label：Clustering result label

%% Calculate H(C1) and H(C2)
N = size(True_label, 1);
k1 = size(unique(True_label), 1);
p = zeros(k1, 1);
H1 = 0;
for i = 1:k1
    n = sum(True_label == i);
    p(i) = n / N;
    H1 = H1 + p(i) * log(p(i));
end
H1 = -H1;

k2 = size(unique(class_label), 1);
q = zeros(k2, 1);
H2 = 0;
for i = 1:k2
    n = sum(class_label == i);
    q(i) = n / N;
    H2 = H2 + q(i) * log(q(i));
end
H2 = -H2;



%% Calculate Mutual Information (MI)
% Calculate contingency table
M = zeros(k1, k2);
for i = 1:N
    M(True_label(i), class_label(i)) = M(True_label(i), class_label(i)) + 1;
end

% Calculate joint probability
P_ij = M ./ N;

% Calculate MI
MI = 0;
for i = 1:k1
    for j = 1:k2
        MI_term = P_ij(i, j) * log(P_ij(i, j) / (p(i) * q(j)));
        if isfinite(MI_term)
            MI = MI + MI_term;
        end
    end
end


%% Calculate the expected Mutual Information E(MI)
% Define a function that computes factorial
    function ret = F(num)
        ret = 1;
        for t = 2:num
            ret = ret + ret * t;
        end
    end

sum_row = sum(M, 2);
sum_col = sum(M, 1);
E_MI = 0;
for i = 1:k1
    for j = 1:k2
        K = max(sum_row(i) + sum_col(j) - N, 1);
        for K = K:min(sum_row(i), sum_col(j))
            E_term = (K / N) * log2((N * K) / (sum_row(i) * sum_col(j))) * ((F(sum_row(i)) * F(sum_col(j)) * F(N - sum_row(i)) * F(N - sum_col(j))) / (F(N) * F(K) * F(sum_row(i) - K) * F(sum_col(j) - K) * F(N - sum_row(i) - sum_col(j) + K)));
            if isfinite(E_term)
                E_MI = E_MI + E_term;
            end
        end
    end
end


%% Calculate Adjusted Mutual Information (AMI)
Ami = (MI - E_MI) / ((H1 + H2) / 2 - E_MI);

end

