function [A,G] = ntd_initialize(Y,init,orthoforce,R)
% Initilization for NTD algorithms
% Output: factors A and core tensor G
% Copyright 2008 of Andrezej Cichocki and Anh Huy Phan
N = ndims(Y);In = size(Y);
if iscell(init)
    if numel(init) ~= N+1
        error('OPTS.init does not have %d cells', N+1);
    end
    for n = 1:N;
        if ~isequal(size(init{n}),[In(n) R(n)])
            error('Factor{%d} is thr wrong size',n);
        end
    end
    if ~isequal(size(init{end}),R)
        error('Core is the wrong size.');
    end
    A = init(1:end-1); G = init{end};
else
    switch init
        case 'random'
            A = arrayfun(@rand,In,R,'uni',0); G = tensor(rand(R));
        case {'nvecs' 'eigs'}
            A = cell(N,1);
            for n = 1:N
                A{n} = nvecs(Y,n,R(n));
            end
            G = ttm(Y, A, 't');
        otherwise
            error('Undefined initialization type');
    end
end
if orthoforce 
    for n = 1:N
        Atilde = ttm(Y, A, -n, 't');
        A{n} = max(eps, nvecs(Atilde,n,R(n)));
    end
    A = cellfun(@(x) bsxfun(@rdivide,x,sum(x)),A,'uni',0);
    G = ttm(Y, A, 't');
end