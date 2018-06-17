function [w, costs, fp_diff, vs, fun, B] = sq_gl_B_fp_acc(X, y, gamma, groups, svdX, svdB, kappa, ...
    w0, iters_fp, eps_fp, iters_acc, eps_acc)

% Minimizes 1/2 ||Xw-y||^2 + gamma ||Bw||_GL
% where ||.||_GL is the group Lasso norm
% using fixed point prox & accelerated
% groups : cell array of group indicators
% svdX : 1 to compute L using SVD of X , 0 to use Frobenius estimate
% svdB : 1 to compute lambda using SVD of B , 0 to use Frobenius estimate

d = size(X,2);
G = length(groups);
ind = [];
cnt = 0;
for i=1:G
  ind{i} = cnt+(1:length(groups{i}));
  cnt = cnt+length(groups{i});
end

f = @(z) (norm((X*z-y),2)^2/2);

    function res = phi(z)
        % Group Lasso norm with varying size groups
        res = 0;
        for i=1:G
            res = res + norm(z(ind{i}));
        end
        res = gamma* res;
%         A = reshape(z,d,length(z)/d);
%         res = gamma * trace(sqrt(A'*A));
    end

gradf = @(z) (X'*(X*z-y));

    function res = prox(z,lam)
        % Group Lasso prox map
        res = [];
        for i=1:G
            nrm = norm(z(ind{i}));
            coeff = max( [ nrm-gamma/lam , 0 ] ,[],2 ) / (nrm+eps);
            res = [res; coeff*z(ind{i})];
        end
        
%         A = reshape(z,d,length(z)/d);
%         nrm = diag(sqrt(A'*A)); 
%         coeff = max( [ nrm-gamma/lam , zeros(length(z)/d,1) ] ,[],2 ) ./ (nrm+eps);
%         res = reshape( (diag(coeff)*A')', length(z),1);
    end

fun = @phi;
% Create group indexing matrix
B = [];
for i=1:length(groups)
    temp = zeros(length(groups{i}),d);
    temp(:, groups{i} ) = eye(length(groups{i}));
    B = [B;temp];
end

if (svdX)
    L = norm(X)^2;
else
    L = sum(X.^2);
end
if (svdB)
  if (size(B,1) > size(B,2))
    lambda = 2*L/ ( norm(B)^2 );
  else
    [U,S] = svd(B);
    lambda = 2*L/ ( max(diag(S))^2 + min(diag(S))^2 );
  end
else
  lambda = 2*L/ ( sum(B.^2) );
end

[w, costs, fp_diff, vs] = fp_acc(f, @phi, B, gradf, L, lambda, kappa, @prox, w0, ...
				 iters_fp, eps_fp, iters_acc, eps_acc);
            
end

