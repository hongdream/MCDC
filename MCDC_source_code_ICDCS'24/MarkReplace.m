function [assignment] = MarkReplace(MarkMat)
%Convert the spatial matrix storing label order into a row vector
[rows,cols]=size(MarkMat);

assignment=zeros(1,cols);

for i=1:rows
    for j=1:cols
        if MarkMat(i,j)==1
            assignment(1,j)=i;
        end
    end
end

end
