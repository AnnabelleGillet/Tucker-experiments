function [ari] = adjusted_rand_index(confusion_matrix)
%adjusted_rand_index 
%   
a = 0;
b = 0;
c = 0;
for i = 1:length(confusion_matrix)
    for j = 1:length(confusion_matrix)
        a = a + ((confusion_matrix(i, j)*(confusion_matrix(i, j) - 1)) / 2);
    end
    b = b + (sum(confusion_matrix(i, :)) * (sum(confusion_matrix(i, :)) - 1) / 2);
    c = c + (sum(confusion_matrix(:, i)) * (sum(confusion_matrix(:, i)) - 1) / 2);
end
d = (sum(sum(confusion_matrix)) * (sum(sum(confusion_matrix)) - 1)) / 2;
ari = (a - b * c / d) / ((1/2) * (b + c) - (b * c / d));
end