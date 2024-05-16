function Acc = ACC(True_label,class_label)
%ACC Accuracy
%   input:
%   True_label： real label of data
%   class_label：Clustering result label
correct = sum(True_label == class_label);
N = size(True_label, 1);
Acc = correct / N;
end

