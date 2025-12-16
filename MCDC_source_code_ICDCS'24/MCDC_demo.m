%% Demo for MCDC
clear;
clc;
%% Data initialization
file = "./Dataset/Che";
data = load(file);
data = struct2cell(data);
data = data{1};
[N, D] = size(data); 
rate = 0.03;            % learning rate
class = data(:, D);     % true label of data
D = D - 1;
data = data(:, 1:D);    % attribute of data
Times = 1;             % experiment times
K = round(sqrt(N));     % initial k
k = size(unique(class), 1);     % true number of clusters of data
result = zeros(Times, 4);
%% MCDC
for T = 1:Times
    %% MGCPL
    [granularity, representation] = MGCPL(data, K, rate);
    D = size(granularity, 2);       % dimensions of representation of data
    class_attribute = zeros(N, D);  % representation of data
    for i = 1:D
        class_attribute(:, i) = representation{i};
    end

    %% GAME
    seed = OI(class_attribute, N, k, D);            % initialization method
    [label, ~] = GAME(k, class_attribute, seed);    
    
    %% Validation
    label = Mapping(class, label);
    result(T, 1) = ACC(label, class);
    result(T, 2) = FM(label, class);
    result(T, 3) = ARI(label, class);
    result(T, 4) = AMI(label, class);
end

%% 输出结果
answer = {"Validation", "Mean", "Var";
          "ACC", sprintf('%.3f', mean(result(:, 1))), sprintf('%.2f', var(result(:, 1)));
          "FM", sprintf('%.3f', mean(result(:, 2))), sprintf('%.2f', var(result(:, 2)));
          "ARI", sprintf('%.3f', mean(result(:, 3))), sprintf('%.2f', var(result(:, 3)));
          "AMI", sprintf('%.3f', mean(result(:, 4))), sprintf('%.2f', var(result(:, 4)));};
disp(answer);
