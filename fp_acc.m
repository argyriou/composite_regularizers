function [x, costs, fp_diff, vs] = fp_acc(f, phi, B, gradf, L, lambda, kappa, prox, x0, ...
			    iters_fp, eps_fp, iters_acc, eps_acc)

% Minimizes regularization functional 
% f(x) + phi(Bx)
% uses Nesterov's accelerated method 
% and Picard iterations to compute the proximity map of phi o B
% B                : m x d matrix
% L                : Lipschitz constant for gradf
% lambda           : has to satisfy ||I-lambda/L * B'B|| <= 1
% kappa            : from Opial's theorem
% prox(x,lambda)   : computes the proximity map of phi/lambda at x
% x0               : initial value
% iters_fp, iters_acc : maximum #iterations for Picard, Nesterov respectively
% eps_fp, eps_acc  : tolerance used as termination criterion in Picard, Nesterov respectively  

t = 1;
alpha = x0;
x = x0;
[m,d] = size(B);
prev_cost = inf;
curr_cost = feval(f,x0) + feval(phi, B*x0);
theta = 1;

while (t < iters_acc && abs(prev_cost - curr_cost) > eps_acc)
  gradalpha = feval(gradf,alpha);
  Bga = B*(gradalpha - L*alpha);
    
  % Fixed point for prox map
  v = zeros(m,1);
  s = 1;
  prev_fp = (eps_fp+100)*ones(m,1);
  while (s <= iters_fp && norm(prev_fp-v) > eps_fp)
    prev_fp = v;
    Bv = B'*v;
    Av = v - lambda/L*B*Bv - 1/L*Bga;
    v = kappa * v + (1-kappa) * (Av - feval(prox,Av,lambda));
    vs{t}{s} = v;
    fp_diff{t}(s) = norm(prev_fp-v);
    s = s+1;
  end
  
  prevx = x;
  x = -1/L* (gradalpha - L*alpha + lambda*B'*v);
  theta = (sqrt(theta^4+4*theta^2)-theta^2)/2;
  rho = 1-theta+sqrt(1-theta);
  alpha = rho*x - (rho-1)*prevx;
  t = t+1;
  prev_cost = curr_cost;
  curr_cost = feval(f,x) + feval(phi, B*x);
  costs(t-1) = curr_cost;
end

