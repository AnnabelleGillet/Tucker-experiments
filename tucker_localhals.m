function [T, A, G, fit, iter] = tucker_localhals(Y, R, opts)
% HALS NTD algorithm
% INPUT
% Y : tensor with size of I1 x I2 x ... x IN
% R : size of core tensor R1 x R2 x ... x RN : [ R1 , R2 , ... , RN ]
% opts : structure of optional parameters for algorithm ( see defoptions )
% .tol : tolerance of stopping criteria ( explained variation ) (1 e−6)
% .maxiters : maximum number of iteration (50)
% .init : initialization type : ' random ', ' eigs ', ' nvecs ' ( HOSVD ) ( random )
% .orthoforce : orthogonal constraint to initialization using ALS
% .ellnorm : normalization type (1)
% .fixsign : fix sign for components of factors. (0)
%
% Copyright 2008 of Anh Huy Phan and Andrzej Cichocki

defoptions = struct('tol',1e-6,'maxiters',50,'init','random','ellnorm',1,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,'fixsign',0);
%defoptions = struct('tol',1e-6,'maxiters',100,'init','eigs','ellnorm',1,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,'fixsign',0);
%defoptions = struct('tol',1e-6,'maxiters',50,'init','nvecs','ellnorm',1,'orthoforce',1,'lda_ortho',0,'lda_smooth',0,'fixsign',0);
if ~exist('opts','var')
    opts = struct;
end
opts = scanparam(defoptions,opts);

% Extract number of dimensions and norm of Y
N = ndims(Y); normY = norm(Y);

if numel(R) == 1, R = R(ones(1,N)); end
if numel(opts.lda_ortho) == 1
    opts.lda_ortho = opts.lda_ortho(ones(1,N));
end
if numel(opts.lda_smooth) == 1
    opts.lda_smooth = opts.lda_smooth(ones(1,N));
end
%% Set up and error checking on initial guess for U.
[A,G] = ntd_initialize(Y,opts.init,opts.orthoforce,R);
%%
fprintf('\nLocal NTD:\n');
% Compute approximate of Y
Yhat = ttm(G,A);
normresidual = norm(Y(:) - Yhat(:));
fit = 1 - (normresidual / normY);
fprintf(' Iter %2d:fit = %e \n', 0, fit);
Yr = Y - Yhat;
%% For smooth constraint
Uf = A;
for n = 1:N
    Uf{n}(:) = 0;
end
%% Main Loop: Iterate until convergence
for iter = 1:opts.maxiters
    pause(0.001)
    fitold = fit;
    
    % Iterate over all N modes of the tensor
    % for smoothness constraint
    for ksm = 1:N
        Uf{ksm} = opts.lda_smooth(ksm) * [A{ksm}(2, :);
            (A{ksm}(1:end-2,:) + A{ksm}(3:end,:))/2; A{ksm}(end-1,:)];
    end
    
    for n = 1:N
        Sn = double(tenmat(ttm(G,A,-n),n));
        Yrn = double(tenmat(Yr,n));
        As = sum(A{n},2);
        for ri = 1:R(n)
            %Ani = A{n}(:,ri) + (Yrn * Sn(ri,:)fl- opts.lda_ortho(n)*(As -A{n}(:,ri)) + Uf{n}(:,ri))/(opts.lda_smooth(n)+Sn(ri,:)*Sn(ri,:)fl);
            Ani = A{n}(:,ri) + (Yrn * Sn(ri,:)'- opts.lda_ortho(n)*(As -A{n}(:,ri)) + Uf{n}(:,ri))/(opts.lda_smooth(n)+Sn(ri,:)*Sn(ri,:)');
            
            if opts.fixsign
                Ani = fixsigns(Ani);
            end
            Ani = max(eps,Ani);
            Yrn = Yrn - (Ani - A{n}(:,ri)) * Sn(ri,:);
            A{n}(:,ri) = Ani;
        end
        A{n} = bsxfun(@rdivide,A{n},sum(A{n}.^opts.ellnorm).^(1/opts.ellnorm));
        Yhat = ttm(G,A);
        Yr = Y - Yhat;
    end
    switch opts.ellnorm
        case 1
            G = G.*ttm(((Y+eps)./(Yhat+eps)),A,'t');
        case 2
            Yr1 = double(tenmat(Yr,1));
            for jgind = 1:prod(R)
                jgsub = ind2sub_full(R,jgind);
                va = arrayfun(@(x) A{x}(:,jgsub(x)),2:N,'uni',0);
                ka = khatrirao(va(end:-1:1));
                gjnew = max(eps,G(jgsub) + A{1}(:,jgsub(1))'*Yr1 * ka);
                Yr1 = Yr1 + (G(jgind) - gjnew) * A{1}(:,jgsub(1))*ka';
                G(jgind) = gjnew;
            end
    end
    Yhat = ttm(G,A);
    if (mod(iter,10) == 1) || (iter == opts.maxiters)
        %Compute fit
        normresidual = sqrt(normY^2 + norm(Yhat)^2 - 2*innerprod(Y,Yhat));
        fit = 1 - (normresidual / normY);
        fitchange = abs(fitold - fit);
        fprintf('Iter %2d: fit = %e delta fit = %7.1e\n',iter,fit,fitchange);
        if (fitchange < opts.tol) && (fit > 0)
            break;
        end
    end
end
T = ttm(G, A);
end