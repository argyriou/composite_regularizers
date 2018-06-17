function [w, costs, fp_diff] = sq_l1_B_fp_acc(X, y, gamma, B, svdX, svdB, kappa, ...
    w0, iters_fp, eps_fp, iters_acc, eps_acc)

% Minimizes 1/2 ||Xw-y||^2 + gamma ||Bw||_1
% using fixed point prox & accelerated
% svdX : 1 to compute L using SVD of X , 0 to use Frobenius estimate
% svdB : 1 to compute lambda using SVD of B , 0 to use Frobenius estimate

f = @(z) (norm((X*z-y),2)^2/2);
phi = @(z) (gamma*norm(z,1));
gradf = @(z) (X'*(X*z-y));
prox = @(z,lam) ( max( [ abs(z)-gamma/lam , zeros(length(z),1) ] ,[],2 ) .* sign(z) );

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

%if (svdB)
%    lambda = 2*L/ ( norm(B)^2 );
%else
%    lambda = 2*L/ ( sum(B.^2) );
%end

[w, costs, fp_diff] = fp_acc(f, phi, B, gradf, L, lambda, kappa, prox, w0, ...
			    iters_fp, eps_fp, iters_acc, eps_acc);
            
end