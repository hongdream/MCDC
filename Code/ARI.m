function [ARI] = ARI(true_label, class_label)
%MY_ARI Adjusted Rand Index
%   ARI = (RI - E(RI)) / (max(RI) - E(RI))

k1 = size(unique(true_label), 1);
k2 = size(unique(class_label), 1);

M = zeros(k1, k2);
N = size(true_label, 1);

%% calculate contingency matrix
for i = 1:N
    M(true_label(i), class_label(i)) = M(true_label(i), class_label(i)) + 1;
end

sum_row = sum(M, 2);
sum_col = sum(M, 1);
a = zeros(1, k1);
for i = 1:k1
    a(i) = nchoosek(sum_row(i), 2);
end

b = zeros(1, k2);
for j = 1:k2
    b(j) = nchoosek(sum_col(j), 2);
end

%% calculate RI
RI = zeros(k1, k2);
for i = 1:k1
    for j = 1:k2
        if M(i, j) < 2
            RI(i, j) = 0;
        else
            RI(i, j) = nchoosek(M(i, j), 2);
        end
    end
end

%% calculate E(RI)
E = sum(a) * sum(b) / nchoosek(N, 2);

%% calculate MAX(RI)
Max = (sum(a) + sum(b)) / 2;

%% calculate ARI
ARI = (sum(RI(:)) - E) / (Max - E);

end

