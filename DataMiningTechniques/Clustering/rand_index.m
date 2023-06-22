function [ri] = rand_index(partition1, partition2)
%rand_index 
%   
a = 0;
b = 0;
total = 0;
for i = 1:length(partition1)
    class_i = partition2(i);
    for j = i + 1:length(partition1)
        total = total + 1;
        class_j = partition2(j);
        if class_i == class_j && partition1(i) == partition1(j)
            a = a + 1;
        elseif class_i ~= class_j && partition1(i) ~= partition1(j)
            b = b + 1;
        end
    end
end
ri = (a + b) / total;
end