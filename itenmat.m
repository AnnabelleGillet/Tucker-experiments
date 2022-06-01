function At = itenmat(A,mode,I)
% Inverse of the tenmat function
% Am = tenmat(A,mode); I = size(A);
% A = itenmat(Am,mode,I);
% Copyright of Anh Huy Phan, Andrzej Cichocki

N = numel(I);
ix = [mode setdiff(1:N,mode)];
At = reshape(A,I(ix));
At = permute(At,[2:mode A mode+1:N]);