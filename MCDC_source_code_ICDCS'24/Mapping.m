function [NewLabel] = Mapping(La1,La2)
% Mapping : Remap labels
% input:
%   True_label: real label of data
%   class_label: Clustering result label
%   NewLabel: mapped labels

Label1=unique(La1');
L1=length(Label1);
Label2=unique(La2');
L2=length(Label2);

%Construct a matrix G that calculates the repeatability of two classification labels
G = zeros(max(L1,L2),max(L1,L2));
for i=1:L1
    index1= La1==Label1(1,i);
    for j=1:L2
        index2= La2==Label2(1,j);
        G(i,j)=sum(index1.*index2);
    end
end

%Use Hungarian algorithm to calculate the matrix after mapping rearrangement
[index]=munkres(-G);
%Convert the map rearrangement result into a row vector storing the label order after the map rearrangement
[temp]=MarkReplace(index);
%Generate the label NewLabel after mapping rearrangement
NewLabel=zeros(size(La2));
for i=1:L2
    NewLabel(La2==Label2(i))=temp(i);
end

end


