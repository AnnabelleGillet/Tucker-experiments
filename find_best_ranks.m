function [ranks] = find_best_ranks(Y, x)
% Find the best rank for each dimension by determining the number of
% singular values needed to represent x% of the sum of all singular values
N = ndims(Y);
r = ones(N, 1);
for n = 1:N
    Yn = double(tenmat(Y,n));
    S = sort(eig(Yn * Yn'), 'descend');
    s = sum(sum(S));
    nb = 0;
    total = 0.0;
    i = 1;
    while total < s * x
        total = total + S(i);
        nb = nb + 1;
        i = i + 1;
    end
    r(n) = nb;
end
ranks = r;
end