function FMI = FM(True_label, class_label)
%FM Fowlkes-Mallows index
% TP：TruePositive
% FP：FalsePositive
% FN：FalseNegative
% FMI = TP / （sqrt（（TP + FP）（TP + FN）））

%% myself_FMI
TP = 0;
FP = 0;
FN = 0;

N = size(True_label, 1);
for i = 1:N
    for j = i + 1:N
        % calculate TP
        if True_label(i) == True_label(j) && class_label(i) == class_label(j)
            TP = TP + 1;
        end
        
        % calculate FP
        if True_label(i) ~= True_label(j) && class_label(i) == class_label(j)
            FP = FP + 1;
        end
        
        % calculate FN
        
        if True_label(i) == True_label(j) && class_label(i) ~= class_label(j)
            FN = FN + 1;
        end
    end
end
FMI = TP / sqrt((TP + FP) * (TP + FN));
