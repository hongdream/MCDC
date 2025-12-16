function U  = OI(x, N, k, d)
%OI Oriented Initialization for Mixed Data Clustering
%   x_categorical : 样本集合的categorical attribute
%   x_numerical : 样本集合的numerical attribute
%   N : 样本个数
%   k : 聚类数量
%   dc : categorical attribute的个数 
    %% 选择u1
    % categorical datas
    sim_c = zeros(N, 1);
    m = max(x(:));
    % 统计每个attribute中 每个值的出现次数
    attribute_frequency = zeros(d, m);
    sample_frequency = zeros(d, 1);
    for i = 1:N
        for r = 1:d
            if x(i, r) ~= 0
                % 每个data，第r个属性值的统计
                attribute_frequency(r, x(i, r)) = attribute_frequency(r, x(i, r)) + 1;
                % 每个data，第r个属性值不为零的统计
                sample_frequency(r, 1) = sample_frequency(r, 1) + 1;
            end
        end
    end
    
    % 计算每个data与整个数据集的similarity
    for i = 1:N
        sim = 0;
        for r = 1:d
            if x(i, r) ~= 0
                sim = sim + attribute_frequency(r, x(i, r)) / sample_frequency(r, 1);
            end
        end
        sim_c(i) = sim ./ d;
    end
    
    % U集合的categorical data统计
    attribute_frequency_U = zeros(d, m);
    sample_frequency_U = zeros(d, 1);
   
    % 选择u1
    cnt = 1;
    sim = sim_c;
    U(cnt) = find(max(sim) == sim, 1);
    
    % 选择剩下的u
    while cnt < k
        Pry_c = zeros(N, 1);
        Pry = zeros(N, 1);
        % categorical datas
        dsim_c = zeros(N, 1);
        for r = 1:d
            if x(U(cnt), r) ~= 0
                attribute_frequency_U(r, x(U(cnt), r)) = attribute_frequency_U(r, x(U(cnt), r)) + 1;
                sample_frequency_U(r, 1) = sample_frequency_U(r, 1) + 1;
            end
        end
        
        for i = 1:N
            dsim = 0;
            for r = 1:d
                if x(i, r) ~= 0
                    dsim = dsim + attribute_frequency_U(r, x(i, r)) / sample_frequency_U(r, 1);
                end
            end
            dsim_c(i) = dsim ./ d;
            Pry_c(i) = (1 - dsim_c(i)) + sim_c(i);
        end
       
        Pry = Pry_c;
        u = find(max(Pry) == Pry, 1);
        cnt = cnt + 1;
        U(cnt) = u;
    end
end

