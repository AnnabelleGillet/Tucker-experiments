function [out] = sort_core_tensor(A)
% Sort the indices of a core tensor depending on the tensor value
[sorted_vals,sorted_idx] = sort(A(:),'descend'); 
c = cell([1 numel(size(A))]); 
[c{:}] = ind2sub(size(A),sorted_idx);
out = [num2cell(sorted_vals) mat2cell([c{:}],ones(1,numel(A)),numel(size(A)))]; 
end